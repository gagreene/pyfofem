# -*- coding: utf-8 -*-
"""
pyfofem.components – Computational sub-models for pyfofem.
"""
from __future__ import annotations

from ._component_helpers import (
    _is_scalar,
    _maybe_scalar,
    _to_str_arr,
)

from .burnup import (
    FuelParticle,
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    burnup,
)

from .soil_heating import (
    soil_heat_campbell,
    soil_heat_massman,
)

from .tree_flame_calcs import (
    SPP_CODES,
    calc_bark_thickness,
    calc_canopy_cover,
    calc_char_ht,
    calc_crown_length_vol_scorched,
    calc_flame_length,
    calc_scorch_ht,
)

from .mortality_calcs import (
    mort_bolchar,
    mort_crnsch,
    mort_crcabe,
)

from .burnup_calcs import (
    gen_burnup_in_file,
    run_burnup,
    _extract_burnup_consumption,
    _burnup_durations,
    _run_burnup_cell,
)

__all__ = [
    # helpers
    '_is_scalar',
    '_maybe_scalar',
    '_to_str_arr',
    # burnup
    'FuelParticle',
    'BurnResult',
    'BurnSummaryRow',
    'BurnupValidationError',
    'burnup',
    'gen_burnup_in_file',
    'run_burnup',
    '_extract_burnup_consumption',
    '_burnup_durations',
    '_run_burnup_cell',
    # soil heating
    'soil_heat_campbell',
    'soil_heat_massman',
    # tree / flame
    'SPP_CODES',
    'calc_bark_thickness',
    'calc_canopy_cover',
    'calc_char_ht',
    'calc_crown_length_vol_scorched',
    'calc_flame_length',
    'calc_scorch_ht',
    # mortality
    'mort_bolchar',
    'mort_crnsch',
    'mort_crcabe',
    # consumption
    'CONSUMPTION_VARS',
    'SOIL_HEAT_VARS',
    'REGION_CODES',
    'CVR_GRP_CODES',
    'SEASON_CODES',
    'FUEL_CATEGORY_CODES',
    'calc_carbon',
    'consm_canopy',
    'consm_duff',
    'consm_herb',
    'consm_litter',
    'consm_mineral_soil',
    'consm_shrub',
    'get_moisture_regime',
    # emissions
    'calc_smoke_emissions',
]
