# -*- coding: utf-8 -*-
"""
pyfofem.py - Facade driver module for FOFEM mortality and consumption/emissions.

This module exposes two top-level driver functions:

* :func:`run_fofem_mortality` - run a FOFEM post-fire tree mortality model.
* :func:`run_fofem_emissions` - run FOFEM fuel consumption, carbon, and
  smoke-emissions modelling.

All component calculations are delegated to the sub-modules in
``pyfofem.components``.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Re-export component symbols for backward compatibility
# ---------------------------------------------------------------------------
from .components.burnup import (
    FuelParticle,
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    burnup as _burnup,
)

from .components._component_helpers import _is_scalar, _maybe_scalar, _to_str_arr

from .components.tree_flame_calcs import (
    SPP_CODES,
    calc_bark_thickness,
    calc_canopy_cover,
    calc_char_ht,
    calc_crown_length_vol_scorched,
    calc_flame_length,
    calc_scorch_ht,
)

from .components.mortality_calcs import (
    mort_bolchar,
    mort_crnsch,
    mort_crcabe,
)

from .components.consumption_calcs import (
    DEFAULT_CONSUMPTION_VARS,
    EXPANDED_CONSUMPTION_VARS,
    TOTAL_DURATION_CONSUMED_VARS,
    SOIL_HEAT_VARS,
    EQUATION_VARS,
    ERROR_VARS,
    REGION_CODES,
    CVR_GRP_CODES,
    SEASON_CODES,
    FUEL_CATEGORY_CODES,
    calc_carbon,
    consm_canopy,
    consm_duff,
    consm_herb,
    consm_litter,
    consm_mineral_soil,
    consm_shrub,
    get_moisture_regime,
    _MOISTURE_REGIMES
)

from .components.burnup_calcs import (
    gen_burnup_in_file,
    run_burnup,
    _extract_burnup_consumption,
    _burnup_durations,
    _run_burnup_cell,
    _DENSITY_SOUND,
    _DENSITY_ROTTEN,
    _SOUND_TPIG,
    _ROTTEN_TPIG,
    _TCHAR,
    _HTVAL,
    _TPAC_TO_KGPM2,
    _KGPM2_TO_TPAC,
    _IN_TO_CM,
)

from .components.burnup import _BURNUP_LIMIT_ADJUST, _BURNUP_LIMIT_ERROR
from .components.soil_heating import soil_heat_campbell

from .components.emission_calcs import (
    _EF_GROUP_DEFAULT,
    _EF_SMOLDERING_GROUP_DEFAULT,
    _EF_DUFF_GROUP_DEFAULT,
    _load_ef_csv,
    calc_smoke_emissions,
)
from .components.emission_pipeline import (
    compute_pre_burnup_consumption,
    initialize_burnup_outputs,
    compute_equation_arrays,
    build_emissions_result,
)


_SOIL_FAMILY_ALIASES = {
    'loamy-skeletal': 'loamy-skeletal',
    'loamy skeletal': 'loamy-skeletal',
    'fine-silty': 'fine-silty',
    'fine-silt': 'fine-silty',
    'fine silt': 'fine-silty',
    'fine': 'fine',
    'coarse-silty': 'coarse-silty',
    'coarse-silt': 'coarse-silty',
    'coarse silt': 'coarse-silty',
    'coarse-loamy': 'coarse-loamy',
    'coarse-loam': 'coarse-loamy',
    'coarse loam': 'coarse-loamy',
}


def _normalize_soil_family(value: str) -> str:
    """Map GUI/C++/user soil family strings to soil_heating.py keys."""
    key = str(value).strip().lower()
    if key in _SOIL_FAMILY_ALIASES:
        return _SOIL_FAMILY_ALIASES[key]
    raise ValueError(
        f"Unrecognised soil_family '{value}'. "
        "Valid options include: Loamy-Skeletal, Fine-Silt(y), Fine, "
        "Coarse-Silt(y), Coarse-Loam(y)."
    )



# ---------------------------------------------------------------------------
# Mortality driver
# ---------------------------------------------------------------------------

# Mapping of mortality function names to callables
_MORT_FUNC_DICT = {
    'bolchar': mort_bolchar,
    'crnsch':  mort_crnsch,
    'crcabe':  mort_crcabe,
}


def run_fofem_mortality(mort_function: str, params: dict) -> Union[float, np.ndarray]:
    """
    Facade driver for FOFEM post-fire tree mortality modelling.

    Selects and calls the appropriate FOFEM mortality function based on the
    *mort_function* argument, forwarding all remaining keyword arguments to
    that function.

    Available mortality functions:

    +-------------+----------------------------------------------+
    | Key         | Model                                        |
    +=============+==============================================+
    | ``bolchar`` | Bole char model (BOLCHAR; Keyser 2018).      |
    |             | For broadleaf hardwood species.              |
    +-------------+----------------------------------------------+
    | ``crnsch``  | Crown scorch model (CRNSCH).                 |
    |             | For conifers and general species.            |
    +-------------+----------------------------------------------+
    | ``crcabe``  | Cambium kill model (CRCABE; Hood & Lutes     |
    |             | 2017). For conifer species only.             |
    +-------------+----------------------------------------------+

    :param mort_function: Name of the mortality sub-model to use.
        One of ``'bolchar'``, ``'crnsch'``, or ``'crcabe'``
        (case-insensitive).
    :param params: Keyword (parameter) arguments forwarded verbatim to the selected
        mortality function.  See each function's own docstring for the full
        parameter list:

        * :func:`~pyfofem.components.mortality_calcs.mort_bolchar`
        * :func:`~pyfofem.components.mortality_calcs.mort_crnsch`
        * :func:`~pyfofem.components.mortality_calcs.mort_crcabe`

    :returns: Mortality probability or array of probabilities (float in
        [0, 1], or ``np.nan`` for unsupported species), matching the return
        type of the selected sub-model.
    :raises KeyError: If *mort_function* is not a recognised key.

    Examples::

        # Crown scorch for a single ponderosa pine tree
        pm = run_fofem_mortality(
            'crnsch',
            spp='PIPO', dbh=25.0, ht=15.0, crown_depth=5.0,
            fire_intensity=500.0,
        )

        # Bole char for multiple broadleaf trees
        pm = run_fofem_mortality(
            'bolchar',
            spp=np.array(['ACRU', 'QUAL']),
            dbh=np.array([12.0, 20.0]),
            char_ht=np.array([1.5, 2.0]),
        )
    """
    key = mort_function.strip().lower()
    if key not in _MORT_FUNC_DICT:
        raise KeyError(
            f"Unknown mortality function '{mort_function}'. "
            f"Valid options: {list(_MORT_FUNC_DICT.keys())}"
        )
    return _MORT_FUNC_DICT[key](**params)


# ---------------------------------------------------------------------------
# Emissions/Consumption driver
# ---------------------------------------------------------------------------

def run_fofem_emissions(
    litter: Union[float, np.ndarray],
    duff: Union[float, np.ndarray],
    duff_depth: Union[float, np.ndarray],
    herb: Union[float, np.ndarray],
    shrub: Union[float, np.ndarray],
    crown_foliage: Union[float, np.ndarray],
    crown_branch: Union[float, np.ndarray],
    pct_crown_burned: Union[float, np.ndarray],
    region: Union[str, int, np.ndarray],
    cvr_grp: Union[str, int, np.ndarray] = '',
    season: Union[str, int, np.ndarray] = 'Summer',
    fuel_category: Union[str, int, np.ndarray] = 'Natural',
    duff_moist: Union[float, np.ndarray, None] = None,
    l_moist: Union[float, np.ndarray, None] = None,
    dw10_moist: Union[float, np.ndarray, None] = None,
    dw1000_moist: Union[float, np.ndarray, None] = None,
    dw1: Union[float, np.ndarray] = 0.0,
    dw10: Union[float, np.ndarray] = 0.0,
    dw100: Union[float, np.ndarray] = 0.0,
    dw1000s: Union[float, np.ndarray] = 0.0,
    dw1000r: Union[float, np.ndarray] = 0.0,
    dw3_6s: Union[float, np.ndarray] = 0.0,
    dw6_9s: Union[float, np.ndarray] = 0.0,
    dw9_20s: Union[float, np.ndarray] = 0.0,
    dw20s: Union[float, np.ndarray] = 0.0,
    dw3_6r: Union[float, np.ndarray] = 0.0,
    dw6_9r: Union[float, np.ndarray] = 0.0,
    dw9_20r: Union[float, np.ndarray] = 0.0,
    dw20r: Union[float, np.ndarray] = 0.0,
    hfi: Optional[Union[float, np.ndarray]] = None,
    flame_res_time: Optional[Union[float, np.ndarray]] = None,
    fuel_bed_depth: Optional[Union[float, np.ndarray]] = None,
    ambient_temp: Optional[Union[float, np.ndarray]] = None,
    windspeed: Optional[Union[float, np.ndarray]] = None,
    use_burnup: bool = True,
    burnup_kwargs: Optional[dict] = None,
    em_mode: str = 'default',
    ef_group: int = _EF_GROUP_DEFAULT,
    ef_csv_path: Optional[str] = None,
    units: str = 'Imperial',
    moisture_regime: Optional[str] = None,
    soil_heating: Union[bool, dict] = False,
    soil_moisture: Optional[Union[float, np.ndarray]] = None,
    soil_family: Optional[Union[str, np.ndarray]] = None,
    num_workers: int = 1,
    show_progress: bool = False,
) -> dict:
    """
    Facade driver for FOFEM fuel consumption, carbon, and smoke emissions.

    All numeric inputs may be scalars or equal-length 1-D ``np.ndarray``\\ s.
    The categorical parameters *region*, *cvr_grp*, *season*, and
    *fuel_category* additionally accept integer codes (see
    :data:`REGION_CODES`, :data:`CVR_GRP_CODES`, :data:`SEASON_CODES`,
    :data:`FUEL_CATEGORY_CODES`) or arrays of strings / integer codes.

    When array inputs are supplied, all non-burnup sub-models are evaluated
    in vectorised numpy operations across all cells.  The burnup model, which
    operates on individual cells, is dispatched in parallel via
    ``concurrent.futures.ProcessPoolExecutor`` with *num_workers* workers
    (one worker → sequential; >1 → parallel across cells).

    :param litter: Pre-fire litter load (T/ac if ``units='Imperial'``,
        kg/m² if ``units='SI'``).
    :param duff: Pre-fire duff load (same units as *litter*).
    :param duff_depth: Pre-fire duff depth (in. if ``units='Imperial'``,
        cm if ``units='SI'``).
    :param herb: Pre-fire herbaceous load (same units as *litter*).
    :param shrub: Pre-fire shrub load (same units as *litter*).
    :param crown_foliage: Pre-fire crown foliage load (same units as *litter*).
    :param crown_branch: Pre-fire crown branch load (same units as *litter*).
    :param pct_crown_burned: Proportion of stand area affected by crown fire (%).
    :param region: FOFEM region. ``str``, ``int`` code (see
        :data:`REGION_CODES`), or ``np.ndarray``.
    :param cvr_grp: FOFEM cover group. ``str``, ``int`` code (see
        :data:`CVR_GRP_CODES`), or ``np.ndarray``. Default ``''``.
    :param season: Burn season. ``str``, ``int`` code (see
        :data:`SEASON_CODES`), or ``np.ndarray``. Default ``'Summer'``.
    :param fuel_category: ``'Natural'``/``1`` or ``'Slash'``/``2`` (see
        :data:`FUEL_CATEGORY_CODES`). Default ``'Natural'``.
    :param duff_moist: Duff gravimetric moisture content (%).
    :param l_moist: Litter (~1-hr) moisture content (%).
    :param dw10_moist: 10-hr woody fuel moisture content (%).
    :param dw1000_moist: 1000-hr woody fuel moisture content (%).
    :param dw1: Pre-fire 1-hr down woody load.
    :param dw10: Pre-fire 10-hr down woody load.
    :param dw100: Pre-fire 100-hr down woody load.
    :param dw1000s: Pre-fire 1000-hr sound down woody load (aggregate).
    :param dw1000r: Pre-fire 1000-hr rotten down woody load (aggregate).
    :param dw3_6s: 1000-hr sound, 3-6 in. diameter.
    :param dw6_9s: 1000-hr sound, 6-9 in. diameter.
    :param dw9_20s: 1000-hr sound, 9-20 in. diameter.
    :param dw20s: 1000-hr sound, >= 20 in. diameter.
    :param dw3_6r: 1000-hr rotten, 3-6 in. diameter.
    :param dw6_9r: 1000-hr rotten, 6-9 in. diameter.
    :param dw9_20r: 1000-hr rotten, 9-20 in. diameter.
    :param dw20r: 1000-hr rotten, >= 20 in. diameter.
    :param hfi: Head fire intensity (kW/m).
    :param flame_res_time: Flame residence time (sec).
    :param fuel_bed_depth: Fuel bed depth (m). Defaults to 0.3 m.
    :param ambient_temp: Ambient air temperature (°C). Defaults to 21 °C.
    :param windspeed: Windspeed at fuelbed top (m/s). Defaults to 0 m/s.
    :param use_burnup: If True (default), run the burnup model.
    :param burnup_kwargs: Optional dict of advanced parameters for
        :func:`run_burnup`.
    :param em_mode: Emission factor mode: ``'legacy'`` (C++ ES_Calc parity),
        ``'default'`` (single EF group), or ``'expanded'`` (flame/coarse/duff groups).
        In ``'legacy'`` mode, smoldering NOx is 0 by design (matching C++).
    :param ef_group: Emission factor group (1-8; default 3).
    :param ef_csv_path: Path to ``emissions_factors.csv`` override.
    :param units: Unit system. ``'Imperial'`` (T/ac, in) or ``'SI'``
        (kg/m², cm).
    :param moisture_regime: Optional named moisture regime. One of
        ``'wet'``, ``'moderate'``, ``'dry'``, or ``'very dry'``
        (case-insensitive). Overrides *duff_moist*, *dw10_moist*,
        *dw1000_moist*, and *l_moist* when provided.
    :param soil_moisture: Optional mineral-soil moisture content (%),
        scalar or array. Used by soil heating when enabled.
    :param soil_heating: ``False`` (default) to skip soil heating;
        ``True`` to run with defaults; or ``dict`` of overrides
        (e.g., ``start_temp``, ``efficiency_wl``, ``efficiency_hs``,
        ``efficiency_duff``).
    :param soil_family: Optional soil family used by the soil-heating model.
        Accepts GUI/C++ names (e.g., ``'Fine-Silt'``) or pyfofem names
        (e.g., ``'fine-silty'``), scalar or array.
    :param num_workers: Number of parallel workers for the burnup loop.
        ``1`` (default) runs sequentially. ``>1`` uses
        ``ProcessPoolExecutor``.
    :param show_progress: If ``True``, display a :mod:`tqdm` progress bar
        during the per-cell burnup loop. Default ``False``.
    :returns: Dict of modeled outputs. Values are plain Python ``float``/``int``
        when all inputs are scalars, otherwise ``np.ndarray``. Expanded
        emission outputs are only included for ``em_mode='legacy'`` or
        ``em_mode='expanded'``; soil-heating outputs are only included when
        ``soil_heating`` is enabled.
    :raises ValueError: If *moisture_regime* is ``None`` and any of the
        four moisture parameters are also ``None``.
    :raises KeyError: If *moisture_regime* or any integer code is not
        recognised.
    """
    import concurrent.futures

    # ------------------------------------------------------------------
    # 0. Moisture regime / guard
    # ------------------------------------------------------------------
    if moisture_regime is not None:
        regime_vals = get_moisture_regime(moisture_regime)
        duff_moist   = regime_vals['duff']
        dw10_moist   = regime_vals['10hr']
        dw1000_moist = regime_vals['3plus']
        l_moist      = regime_vals['10hr'] - 2.0
    else:
        if dw10_moist is None or dw1000_moist is None or duff_moist is None:
            raise ValueError('Missing at least one required moisture input: dw10_moist, dw1000_moist, duff_moist. '
                             'Either provide these explicitly, or specify a moisture_regime.')
        if l_moist is None:
            l_moist = dw10_moist - 2.0
        _missing = [name for name, val in (
            ('duff_moist',   duff_moist),
            ('l_moist',      l_moist),
            ('dw10_moist',   dw10_moist),
            ('dw1000_moist', dw1000_moist),
        ) if val is None]
        if _missing:
            raise ValueError(
                f"moisture_regime was not provided, so the following moisture "
                f"parameters are required but were not supplied: {_missing}. "
                f"Either pass explicit values for each, or supply a "
                f"moisture_regime (one of {list(_MOISTURE_REGIMES.keys())})."
            )

    # ------------------------------------------------------------------
    # 1. Detect scalar vs. array call; broadcast all numeric inputs
    # ------------------------------------------------------------------
    _scalar_inputs = [
        litter, duff, duff_depth, herb, shrub, crown_foliage, crown_branch,
        pct_crown_burned, duff_moist, l_moist, dw10_moist, dw1000_moist,
        dw1, dw10, dw100, dw1000s, dw1000r,
        dw3_6s, dw6_9s, dw9_20s, dw20s, dw3_6r, dw6_9r, dw9_20r, dw20r,
    ]
    # Also check categorical params for scalar-ness
    _cat_scalar = all(
        not isinstance(v, np.ndarray)
        for v in (region, cvr_grp, season, fuel_category, soil_family)
    )
    scalar_call = _cat_scalar and all(_is_scalar(v) for v in _scalar_inputs) and _is_scalar(soil_moisture)

    # Broadcast numeric inputs to a common 1-D array shape
    arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, dtype=float)) for v in _scalar_inputs])
    (
        lit_a, duf_a, duf_dep_a, her_a, shr_a, fol_a, bra_a,
        pcb_a, duf_m_a, l_m_a, dw10_m_a, dw1k_m_a,
        dw1_a, dw10_a, dw100_a, dw1ks_agg_a, dw1kr_agg_a,
        s3_6_a, s6_9_a, s9_20_a, s20_a, r3_6_a, r6_9_a, r9_20_a, r20_a,
    ) = arrs
    n = len(lit_a)

    # Optional fire environment arrays (default to broadcast scalar)
    hfi_a   = np.broadcast_to(np.atleast_1d(np.asarray(hfi,   dtype=float)) if hfi   is not None else np.full(n, np.nan), (n,)).copy()
    frt_a   = np.broadcast_to(np.atleast_1d(np.asarray(flame_res_time, dtype=float)) if flame_res_time is not None else np.full(n, np.nan), (n,)).copy()
    fb_a    = np.broadcast_to(np.atleast_1d(np.asarray(fuel_bed_depth, dtype=float)) if fuel_bed_depth is not None else np.full(n, 0.3), (n,)).copy()
    at_a    = np.broadcast_to(np.atleast_1d(np.asarray(ambient_temp,   dtype=float)) if ambient_temp   is not None else np.full(n, 27.0), (n,)).copy()
    ws_a    = np.broadcast_to(np.atleast_1d(np.asarray(windspeed,      dtype=float)) if windspeed      is not None else np.full(n, 0.0), (n,)).copy()

    # Categorical arrays
    reg_a  = _to_str_arr(region,        REGION_CODES)
    cvr_a  = _to_str_arr(cvr_grp,       CVR_GRP_CODES)
    sea_a  = _to_str_arr(season,        SEASON_CODES)
    ft_a   = _to_str_arr(fuel_category, FUEL_CATEGORY_CODES)
    reg_a  = np.broadcast_to(reg_a,  (n,)) if reg_a.size  == 1 else reg_a
    cvr_a  = np.broadcast_to(cvr_a,  (n,)) if cvr_a.size  == 1 else cvr_a
    sea_a  = np.broadcast_to(sea_a,  (n,)) if sea_a.size  == 1 else sea_a
    ft_a   = np.broadcast_to(ft_a,   (n,)) if ft_a.size   == 1 else ft_a

    soil_cfg: Dict[str, Any]
    if isinstance(soil_heating, dict):
        soil_cfg = dict(soil_heating)
        soil_enabled = bool(soil_cfg.get('enabled', True))
    else:
        soil_cfg = {}
        soil_enabled = bool(soil_heating)

    if soil_enabled and soil_family is None:
        raise ValueError(
            "soil_heating requested but soil_family was not provided."
        )

    soil_family_a = np.full(n, '', dtype=object)
    soil_family_valid = np.zeros(n, dtype=bool)
    if soil_family is None:
        pass
    elif isinstance(soil_family, np.ndarray):
        sf_arr = soil_family.ravel()
        if sf_arr.size == 1:
            sf_arr = np.full(n, sf_arr[0], dtype=object)
        elif sf_arr.size != n:
            raise ValueError(
                f"soil_family array length ({sf_arr.size}) must equal number of cells ({n}) or be scalar."
            )
        if soil_enabled:
            for i, v in enumerate(sf_arr):
                try:
                    soil_family_a[i] = _normalize_soil_family(v)
                    soil_family_valid[i] = True
                except ValueError:
                    # Invalid per-cell soil families are skipped in soil-heating.
                    continue
        else:
            soil_family_a = np.array([_normalize_soil_family(v) for v in sf_arr], dtype=object)
            soil_family_valid[:] = True
    else:
        sf_val = str(soil_family)
        if soil_enabled:
            try:
                sf_norm = _normalize_soil_family(sf_val)
                soil_family_a[:] = sf_norm
                soil_family_valid[:] = True
            except ValueError:
                # Invalid scalar family skips soil-heating for all cells.
                pass
        else:
            sf_norm = _normalize_soil_family(sf_val)
            soil_family_a[:] = sf_norm
            soil_family_valid[:] = True

    if moisture_regime is not None:
        soil_moist_default = float(regime_vals['soil'])
    else:
        # Fallback handled per-cell from duf_m_a (clipped to 0..25).
        soil_moist_default = float("nan")

    soil_start_temp = float(soil_cfg.get('start_temp', 21.0))
    soil_moist_override = soil_cfg.get('soil_moisture', None)
    if soil_moisture is not None:
        _sm = np.atleast_1d(np.asarray(soil_moisture, dtype=float))
        if _sm.size == 1:
            soil_moist_arr = np.full(n, float(_sm[0]), dtype=float)
        elif _sm.size == n:
            soil_moist_arr = _sm.astype(float)
        else:
            raise ValueError(
                f"soil_moisture length ({_sm.size}) must equal number of cells ({n}) or be scalar."
            )
    else:
        soil_moist_arr = None
    eff_wl_default = float(soil_cfg.get('efficiency_wl', 0.15))
    eff_hs_default = float(soil_cfg.get('efficiency_hs', 0.10))
    eff_duff_default = float(soil_cfg.get('efficiency_duff', 1.0))
    depth_layers = [float(d) for d in soil_cfg.get('depth_layers_cm', list(range(1, 14)))]
    if len(depth_layers) != 13:
        raise ValueError("soil_heating depth_layers_cm must contain exactly 13 depths.")

    is_imperial = units.strip().lower() in ('imperial', 'english')
    to_si  = _TPAC_TO_KGPM2 if is_imperial else 1.0
    from_si = 1.0 / to_si

    bkw_base = {
        'r0': 1.83, 'dr': 0.4, 'timestep': 15.0, 'max_times': 3000,
        'fint_switch': 15.0, 'validate': True,
    }
    if burnup_kwargs:
        bkw_base.update(burnup_kwargs)
    burnup_dt = float(bkw_base['timestep'])

    # ------------------------------------------------------------------
    # 2. Resolve 1000-hr class loads
    # ------------------------------------------------------------------
    # If per-size-class arrays are zero but aggregate is nonzero, dump aggregate into 3-6 bin
    dw1ks_agg = dw1ks_agg_a
    dw1kr_agg = dw1kr_agg_a
    s3_6  = np.where((s3_6_a + s6_9_a + s9_20_a + s20_a == 0) & (dw1ks_agg > 0), dw1ks_agg, s3_6_a)
    s6_9  = s6_9_a.copy();  s9_20 = s9_20_a.copy();  s20 = s20_a.copy()
    r3_6  = np.where((r3_6_a + r6_9_a + r9_20_a + r20_a == 0) & (dw1kr_agg > 0), dw1kr_agg, r3_6_a)
    r6_9  = r6_9_a.copy();  r9_20 = r9_20_a.copy();  r20 = r20_a.copy()

    dw1ks_pre = s3_6 + s6_9 + s9_20 + s20
    dw1kr_pre = r3_6 + r6_9 + r9_20 + r20

    # ------------------------------------------------------------------
    # 3. Season capitalise
    # ------------------------------------------------------------------
    sea_a = np.array([s.capitalize() for s in sea_a], dtype=object)

    # ------------------------------------------------------------------
    # 4. Vectorized pre-burnup consumption and default output state
    # ------------------------------------------------------------------
    pre = compute_pre_burnup_consumption(
        lit_a=lit_a, l_m_a=l_m_a, cvr_a=cvr_a, reg_a=reg_a, units=units,
        her_a=her_a, sea_a=sea_a, shr_a=shr_a, pcb_a=pcb_a,
        fol_a=fol_a, bra_a=bra_a, ft_a=ft_a, duf_m_a=duf_m_a, duf_a=duf_a,
        duf_dep_a=duf_dep_a, dw10_a=dw10_a, dw1_a=dw1_a, dw1k_m_a=dw1k_m_a,
    )
    lit_pre_arr = pre['lit_pre_arr']; lit_con_arr = pre['lit_con_arr']; lit_pos_arr = pre['lit_pos_arr']
    her_pre_arr = pre['her_pre_arr']; her_con_arr = pre['her_con_arr']; her_pos_arr = pre['her_pos_arr']
    shr_pre_arr = pre['shr_pre_arr']; shr_con_arr = pre['shr_con_arr']; shr_pos_arr = pre['shr_pos_arr']
    fol_pre_arr = pre['fol_pre_arr']; fol_con_arr = pre['fol_con_arr']; fol_pos_arr = pre['fol_pos_arr']
    bra_pre_arr = pre['bra_pre_arr']; bra_con_arr = pre['bra_con_arr']; bra_pos_arr = pre['bra_pos_arr']
    mse_arr = pre['mse_arr']
    duf_pre_arr = pre['duf_pre_arr']; duf_dep_pre_arr = pre['duf_dep_pre_arr']
    duf_dep_con_arr = pre['duf_dep_con_arr']; duf_dep_pos_arr = pre['duf_dep_pos_arr']
    pdc_arr = pre['pdc_arr']

    init = initialize_burnup_outputs(
        n=n, duf_pre_arr=duf_pre_arr, pdc_arr=pdc_arr, dw1_a=dw1_a,
        dw10_a=dw10_a, dw100_a=dw100_a, dw10_m_a=dw10_m_a, dw1k_m_a=dw1k_m_a,
        dw1ks_pre=dw1ks_pre, dw1kr_pre=dw1kr_pre, lit_con_arr=lit_con_arr, frt_a=frt_a,
    )
    duf_con_arr = init['duf_con_arr']; duf_pos_arr = init['duf_pos_arr']
    dw1_pre_arr = init['dw1_pre_arr']; dw1_con_arr = init['dw1_con_arr']; dw1_pos_arr = init['dw1_pos_arr']
    dw10_pre_arr = init['dw10_pre_arr']; dw10_con_arr = init['dw10_con_arr']; dw10_pos_arr = init['dw10_pos_arr']
    dw100_pre_arr = init['dw100_pre_arr']; dw100_con_arr = init['dw100_con_arr']; dw100_pos_arr = init['dw100_pos_arr']
    dw1ks_con_arr = init['dw1ks_con_arr']; dw1ks_pos_arr = init['dw1ks_pos_arr']
    dw1kr_con_arr = init['dw1kr_con_arr']; dw1kr_pos_arr = init['dw1kr_pos_arr']
    snd_fla_arr = init['snd_fla_arr']; snd_smo_arr = init['snd_smo_arr']
    rot_fla_arr = init['rot_fla_arr']; rot_smo_arr = init['rot_smo_arr']
    fine_fla_arr = init['fine_fla_arr']; fine_smo_arr = init['fine_smo_arr']
    lit_fla_arr = init['lit_fla_arr']; lit_smo_arr = init['lit_smo_arr']
    fla_dur_arr = init['fla_dur_arr']; smo_dur_arr = init['smo_dur_arr']
    burnup_adj_arr = init['burnup_adj_arr']; burnup_err_arr = init['burnup_err_arr']; burnup_ran = init['burnup_ran']
    lay0_arr = init['lay0_arr']; lay2_arr = init['lay2_arr']; lay4_arr = init['lay4_arr']
    lay6_arr = init['lay6_arr']; lay60d_arr = init['lay60d_arr']; lay275d_arr = init['lay275d_arr']

    # ------------------------------------------------------------------
    # 5. Per-cell burnup (parallelised)
    # ------------------------------------------------------------------
    burnup_times_cells = [None] * n
    burnup_wl_cells = [None] * n
    burnup_hs_cells = [None] * n

    if use_burnup:
        # Build per-cell kwargs list
        _DW1HR_ADJ = 0.02; _DW100HR_ADJ = 0.02
        _DW1000HR_ADJ_ROT = 2.5; _BURNUP_MOIST_UPPER = 3.0

        cell_kwargs = []
        for i in range(n):
            d10m  = float(dw10_m_a[i])
            d1km  = float(dw1k_m_a[i])
            d10f  = d10m / 100.0
            d1kf  = d1km / 100.0
            # Rotten 1000-hr moisture content = min(sound 1000-hr moisture content * 2.5, 300%)
            drotf = min(d1kf * _DW1000HR_ADJ_ROT, _BURNUP_MOIST_UPPER)

            fl: Dict[str, float] = {}
            fm: Dict[str, float] = {}
            rk: Dict[str, str]   = {}
            dm: Dict[str, float] = {}

            def _add(key, val, moist):
                v = float(val) * to_si
                if v > 0:
                    fl[key] = v
                    fm[key] = max(moist, 0.02)

            _add('litter', lit_pre_arr[i], max(d10f - _DW1HR_ADJ, 0.02))
            _add('dw1',    dw1_pre_arr[i],  max(d10f - _DW1HR_ADJ, 0.02))
            _add('dw10',   dw10_pre_arr[i], max(d10f, 0.02))
            _add('dw100',  dw100_pre_arr[i], max(d10f + _DW100HR_ADJ, 0.02))
            _add('dwk_3_6',  float(s3_6[i]),  max(d1kf, 0.02))
            _add('dwk_6_9',  float(s6_9[i]),  max(d1kf, 0.02))
            _add('dwk_9_20', float(s9_20[i]), max(d1kf, 0.02))
            _add('dwk_20',   float(s20[i]),   max(d1kf, 0.02))

            for rkey, skey in (('dwk_3_6_r','dwk_3_6'),('dwk_6_9_r','dwk_6_9'),
                                ('dwk_9_20_r','dwk_9_20'),('dwk_20_r','dwk_20')):
                rval_map = {'dwk_3_6_r': r3_6, 'dwk_6_9_r': r6_9,
                            'dwk_9_20_r': r9_20, 'dwk_20_r': r20}
                v = float(rval_map[rkey][i]) * to_si
                if v > 0:
                    fl[rkey] = v; fm[rkey] = max(drotf, 0.02)
                    rk[rkey] = skey; dm[rkey] = _DENSITY_ROTTEN

            duf_si   = float(duf_pre_arr[i]) * to_si
            # C++ BCM_SetInputs: if DW1 load is zero, inject 0.0000001 kg/m²
            # so burnup always has at least one fuel particle. This lets
            # burnup handle duff-only burns correctly.
            if not fl and duf_si > 0:
                fl['dw1'] = 1e-7  # matches C++ BCM_SetInputs guard
                fm['dw1'] = max(d10f - _DW1HR_ADJ, 0.02)
            duf_mf   = max(float(duf_m_a[i]) / 100.0, 0.02) if duf_si > 0 else 2.0
            hfi_val  = float(hfi_a[i]) if not np.isnan(hfi_a[i]) else None
            frt_val  = float(frt_a[i]) if not np.isnan(frt_a[i]) else (60.0 if (hfi_val or 0) > 0 else None)
            intensity = hfi_val if hfi_val is not None else 50.0
            ig_time   = frt_val if frt_val is not None else 60.0

            # Herb+shrub and branch+foliage consumed (kg/m²) for burnup
            # fire-intensity contribution — mirrors C++ BRN_Run / HSB / BRN_Intensity.
            _her_v = float(her_con_arr[i]) if not np.isnan(her_con_arr[i]) else 0.0
            _shr_v = float(shr_con_arr[i]) if not np.isnan(shr_con_arr[i]) else 0.0
            hsf_si = (_her_v + _shr_v) * to_si
            brafol_si = (float(fol_con_arr[i]) + float(bra_con_arr[i])) * to_si

            cell_kwargs.append({
                'fuel_loadings_bu': fl,
                'fuel_moistures_bu': fm,
                'rotten_keys': rk,
                'density_map': dm,
                'intensity_kw': intensity,
                'frt_s': ig_time,
                'ws': float(ws_a[i]),
                'fb_depth': float(fb_a[i]),
                'amb_temp': float(at_a[i]),
                'duf_loading_si': duf_si,
                'duf_moist_frac': duf_mf,
                'duf_pct_consumed': float(pdc_arr[i]),  # FOFEM DUF_Mngr pdc → DuffBurn ff
                'hsf_consumed_si': hsf_si,
                'brafol_consumed_si': brafol_si,
                'burnup_dt': burnup_dt,
                'bkw': bkw_base,
                'cell_idx': i,
            })

        # Run burnup cells using the top-level (picklable) worker function
        chunk_size = 50
        _tqdm_kw = dict(total=len(cell_kwargs),
                        desc='Burnup',
                        unit='cells',
                        disable=not show_progress)
        if num_workers == 1:
            from tqdm import tqdm
            cell_results = [
                _run_burnup_cell(ck)
                for ck in tqdm(cell_kwargs, **_tqdm_kw)
            ]
        else:
            from tqdm import tqdm
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
                cell_results = list(tqdm(
                    pool.map(_run_burnup_cell, cell_kwargs, chunksize=chunk_size),
                    **_tqdm_kw,
                ))

        # Merge burnup results back into output arrays
        for i, cr in enumerate(cell_results):
            if cr is None:
                continue
            burnup_adj_arr[i] = cr.get('burnup_limit_adjust', 0)
            burnup_err_arr[i] = cr.get('burnup_error', 0)

            # If burnup errored, skip consumption merge (use simplified defaults)
            if cr.get('burnup_error', 0) != 0:
                continue
            if 'bcon' not in cr:
                continue

            bcon = cr['bcon']
            burnup_times_cells[i] = cr.get('burnup_times_s')
            burnup_wl_cells[i] = cr.get('burnup_fi_wl')
            burnup_hs_cells[i] = cr.get('burnup_fi_hs')
            fsi  = from_si
            burnup_ran[i] = True

            if 'litter' in bcon:
                # C++ BCM_Mngr: for SouthEast and Pine Flatwoods, the
                # regional litter equation (998/997) overrides burnup's
                # litter consumed. Burnup still runs with the litter for
                # fire intensity, but the consumed amount comes from the
                # regional equation computed earlier in step 4.
                _fw = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd')
                _is_se_lit_override = (
                    str(reg_a[i]) == 'SouthEast' or
                    str(cvr_a[i]) in _fw
                )
                if not _is_se_lit_override:
                    lit_con_arr[i] = bcon['litter']['consumed'] * fsi
                    lit_pos_arr[i] = lit_pre_arr[i] - lit_con_arr[i]
                lit_fla_arr[i] = bcon['litter']['flaming'] * fsi
                lit_smo_arr[i] = bcon['litter']['smoldering'] * fsi

            dw1_con_arr[i]  = bcon['dw1']['consumed'] * fsi  if 'dw1'  in bcon else dw1_pre_arr[i]
            dw1_pos_arr[i]  = dw1_pre_arr[i] - dw1_con_arr[i]
            dw10_con_arr[i] = bcon['dw10']['consumed'] * fsi if 'dw10' in bcon else dw10_pre_arr[i]
            dw10_pos_arr[i] = dw10_pre_arr[i] - dw10_con_arr[i]

            if 'dw100' in bcon:
                dw100_con_arr[i] = bcon['dw100']['consumed'] * fsi
            dw100_pos_arr[i] = dw100_pre_arr[i] - dw100_con_arr[i]

            sc = sm = rc = rm = 0.0
            for k in ('dwk_3_6','dwk_6_9','dwk_9_20','dwk_20'):
                if k in bcon:
                    sc += bcon[k]['consumed']   * fsi
                    sm += bcon[k]['smoldering'] * fsi
                    snd_fla_arr[i] += bcon[k]['flaming'] * fsi
            for k in ('dwk_3_6_r','dwk_6_9_r','dwk_9_20_r','dwk_20_r'):
                if k in bcon:
                    rc += bcon[k]['consumed']   * fsi
                    rm += bcon[k]['smoldering'] * fsi
                    rot_fla_arr[i] += bcon[k]['flaming'] * fsi
            dw1ks_con_arr[i] = sc;  dw1ks_pos_arr[i] = dw1ks_pre[i] - sc
            snd_smo_arr[i]   = sm
            dw1kr_con_arr[i] = rc;  dw1kr_pos_arr[i] = dw1kr_pre[i] - rc
            rot_smo_arr[i]   = rm

            ff = fm2 = 0.0
            for k in ('dw1','dw10','dw100'):  # litter is tracked in lit_fla_arr/lit_smo_arr
                if k in bcon:
                    ff  += bcon[k]['flaming']    * fsi
                    fm2 += bcon[k]['smoldering'] * fsi
            fine_fla_arr[i] = ff;  fine_smo_arr[i] = fm2

            fla_dur_arr[i] = cr['fla_dur']
            smo_dur_arr[i] = cr['smo_dur']
            # NOTE: duf_con_arr / duf_pos_arr are intentionally NOT updated here.
            # Duff load consumed is determined entirely by DUF_Mngr's pdc (computed
            # in the per-cell consm_duff loop above). This matches C++ behaviour where
            # DuffBurn uses f_DufConPerCent from DUF_Mngr and burnup only
            # controls timing/intensity, not the total consumed amount.

    # ------------------------------------------------------------------
    # 5b. Zero out all per-cell outputs for cells with a burnup error
    # ------------------------------------------------------------------
    if use_burnup:
        err_mask = burnup_err_arr != 0
        if np.any(err_mask):
            for arr in (
                lit_con_arr, lit_pos_arr, lit_fla_arr, lit_smo_arr,
                her_con_arr, her_pos_arr,
                shr_con_arr, shr_pos_arr,
                fol_con_arr, fol_pos_arr,
                bra_con_arr, bra_pos_arr,
                duf_con_arr, duf_pos_arr,
                duf_dep_con_arr, duf_dep_pos_arr,
                dw1_con_arr, dw1_pos_arr,
                dw10_con_arr, dw10_pos_arr,
                dw100_con_arr, dw100_pos_arr,
                dw1ks_con_arr, dw1ks_pos_arr,
                dw1kr_con_arr, dw1kr_pos_arr,
                snd_fla_arr, snd_smo_arr,
                rot_fla_arr, rot_smo_arr,
                fine_fla_arr, fine_smo_arr,
                mse_arr,
            ):
                arr[err_mask] = 0.0
            fla_dur_arr[err_mask] = 0.0
            smo_dur_arr[err_mask] = 0.0

    # ------------------------------------------------------------------
    # 6. Smoldering duration fallback where burnup didn't provide it
    # ------------------------------------------------------------------
    for i in range(n):
        if burnup_err_arr[i] != 0:
            continue
        # C++ DuffBurn parity for duff-only cases.
        # When there are no non-duff fuels, C++ duration tracks DuffBurn tdf:
        #   tdf = 1e4 * ff * wdf / (7.5 - 2.7 * dfm)
        # where ff is duff fraction consumed and wdf is kg/m^2.
        _non_duff_pre = (
            lit_pre_arr[i] + dw1_pre_arr[i] + dw10_pre_arr[i] + dw100_pre_arr[i] +
            dw1ks_pre[i] + dw1kr_pre[i] + her_pre_arr[i] + shr_pre_arr[i] +
            fol_pre_arr[i] + bra_pre_arr[i]
        )
        if _non_duff_pre <= 0.0 and duf_con_arr[i] > 0.0:
            _dfm = float(duf_m_a[i]) / 100.0
            _wdf = float(duf_pre_arr[i]) * to_si if is_imperial else float(duf_pre_arr[i])
            _ff = float(pdc_arr[i]) / 100.0 if 0.0 <= float(pdc_arr[i]) <= 100.0 else (0.837 - 0.426 * _dfm)
            _den = 7.5 - 2.7 * _dfm
            if _wdf > 0.0 and _dfm < 1.96 and _den > 0.0 and _ff > 0.0:
                smo_dur_arr[i] = 1.0e4 * _ff * _wdf / _den
                continue
        if np.isnan(smo_dur_arr[i]) and not np.isnan(duf_dep_con_arr[i]) and duf_dep_con_arr[i] > 0:
            dep_cm  = duf_dep_con_arr[i] * _IN_TO_CM if is_imperial else duf_dep_con_arr[i]
            brate   = max(0.05 * np.exp(-0.025 * float(duf_m_a[i])), 1e-6)
            smo_dur_arr[i] = (dep_cm / brate) * 60.0

    # ------------------------------------------------------------------
    # 6b. Soil heating (FOFEM SH_Mngr parity path)
    # ------------------------------------------------------------------
    if soil_enabled:
        def _cfg_float_at(key: str, default: float, idx: int) -> float:
            val = soil_cfg.get(key, default)
            if isinstance(val, np.ndarray):
                arr = val.ravel()
                if arr.size == 1:
                    return float(arr[0])
                if arr.size == n:
                    return float(arr[idx])
                raise ValueError(f"soil_heating['{key}'] length must be 1 or {n}.")
            if isinstance(val, (list, tuple)):
                if len(val) == 1:
                    return float(val[0])
                if len(val) == n:
                    return float(val[idx])
                raise ValueError(f"soil_heating['{key}'] length must be 1 or {n}.")
            return float(val)

        for i in range(n):
            if burnup_err_arr[i] != 0:
                continue
            if not soil_family_valid[i]:
                continue

            if soil_moist_arr is not None:
                soil_moist_i = float(soil_moist_arr[i])
            elif soil_moist_override is not None:
                soil_moist_i = _cfg_float_at('soil_moisture', soil_moist_default, i)
            elif moisture_regime is not None:
                soil_moist_i = soil_moist_default
            else:
                soil_moist_i = float(np.clip(duf_m_a[i], 0.0, 25.0))
            soil_params = {
                'soil_family': str(soil_family_a[i]),
                'start_water': float(np.clip(soil_moist_i / 100.0, 0.0, 1.0)),
                'start_temp': _cfg_float_at('start_temp', soil_start_temp, i),
            }

            duf_depth_pre_in = float(duf_dep_pre_arr[i]) if is_imperial else float(duf_dep_pre_arr[i]) / 2.54
            duf_load_pre_tac = float(duf_pre_arr[i]) if is_imperial else float(duf_pre_arr[i]) * _KGPM2_TO_TPAC
            duf_pct = float(pdc_arr[i])
            duf_moist_pct = float(duf_m_a[i])

            model = 'duff' if duf_depth_pre_in > 0.0 else 'non_duff'
            try:
                if model == 'duff':
                    df_soil = soil_heat_campbell(
                        model='duff',
                        duff_params={
                            'duff_load': duf_load_pre_tac,
                            'duff_depth': duf_depth_pre_in,
                            'duff_moisture': duf_moist_pct,
                            'pct_consumed': duf_pct,
                            'efficiency_duff': _cfg_float_at('efficiency_duff', eff_duff_default, i),
                        },
                        soil_params=soil_params,
                        depth_layers=depth_layers,
                        timestep=_cfg_float_at('timestep_s', 10.0, i),
                    )
                else:
                    wl_series = burnup_wl_cells[i]
                    hs_series = burnup_hs_cells[i]
                    t_series = burnup_times_cells[i]
                    if not wl_series or not t_series:
                        fi_fallback = float(hfi_a[i]) if not np.isnan(hfi_a[i]) else 20.0
                        t_fallback = float(frt_a[i]) if not np.isnan(frt_a[i]) else 60.0
                        wl_series = [max(fi_fallback, 0.0)]
                        hs_series = [0.0]
                        t_series = [max(t_fallback, 1.0)]

                    df_soil = soil_heat_campbell(
                        model='non_duff',
                        duff_params={},
                        soil_params=soil_params,
                        depth_layers=depth_layers,
                        burnup_intensity=wl_series,
                        burnup_intensity_hs=hs_series,
                        burnup_times=t_series,
                        efficiency_wl=_cfg_float_at('efficiency_wl', eff_wl_default, i),
                        efficiency_hs=_cfg_float_at('efficiency_hs', eff_hs_default, i),
                        timestep=_cfg_float_at('timestep_s', 10.0, i),
                    )

                max_t = df_soil.max(axis=0)
                lay0_arr[i] = float(max_t.get('Surface', np.nan))
                lay2_arr[i] = float(max_t.get('2cm', np.nan))
                lay4_arr[i] = float(max_t.get('4cm', np.nan))
                lay6_arr[i] = float(max_t.get('6cm', np.nan))

                lay_cols = ['Surface'] + [f"{int(d)}cm" for d in depth_layers]
                lay60 = -1
                lay275 = -1
                for lay_idx, col in enumerate(lay_cols):
                    if col in df_soil and bool((df_soil[col] > 60.0).any()):
                        lay60 = lay_idx
                    if col in df_soil and bool((df_soil[col] > 275.0).any()):
                        lay275 = lay_idx
                lay60d_arr[i] = float(lay60)
                lay275d_arr[i] = float(lay275)
            except Exception:
                # Preserve pipeline robustness: failed soil model should not
                # invalidate consumption/emissions outputs.
                continue

    # ------------------------------------------------------------------
    # 7. Flaming / smoldering totals
    # ------------------------------------------------------------------
    fla_con_arr = (lit_fla_arr + fine_fla_arr + her_con_arr + shr_con_arr
                   + fol_con_arr + bra_con_arr + snd_fla_arr + rot_fla_arr)
    smo_con_arr = (duf_con_arr + lit_smo_arr + fine_smo_arr
                   + snd_smo_arr + rot_smo_arr)

    # ------------------------------------------------------------------
    # 8. Equation number arrays
    # ------------------------------------------------------------------
    lit_eq_arr, duf_eq_arr, herb_eq_arr, shrub_eq_arr = compute_equation_arrays(reg_a, cvr_a)

    # ------------------------------------------------------------------
    # 9. Smoke emissions (vectorised call)
    # ------------------------------------------------------------------
    duff_load_for_emissions = duf_con_arr if em_mode in ('legacy', 'expanded') else 0.0
    emissions = calc_smoke_emissions(
        flaming_load=fla_con_arr,
        smoldering_load=smo_con_arr,
        mode=em_mode,
        ef_group=ef_group,
        duff_load=duff_load_for_emissions,
        ef_csv_path=ef_csv_path,
        units=units,
    )

    # ------------------------------------------------------------------
    # 10. Build final output
    # ------------------------------------------------------------------
    return build_emissions_result(
        scalar_call=scalar_call,
        soil_enabled=soil_enabled,
        em_mode=em_mode,
        expanded_consumption_vars=set(EXPANDED_CONSUMPTION_VARS),
        emissions=emissions,
        lit_pre_arr=lit_pre_arr, lit_con_arr=lit_con_arr, lit_pos_arr=lit_pos_arr,
        dw1_pre_arr=dw1_pre_arr, dw1_con_arr=dw1_con_arr, dw1_pos_arr=dw1_pos_arr,
        dw10_pre_arr=dw10_pre_arr, dw10_con_arr=dw10_con_arr, dw10_pos_arr=dw10_pos_arr,
        dw100_pre_arr=dw100_pre_arr, dw100_con_arr=dw100_con_arr, dw100_pos_arr=dw100_pos_arr,
        dw1ks_pre=dw1ks_pre, dw1ks_con_arr=dw1ks_con_arr, dw1ks_pos_arr=dw1ks_pos_arr,
        dw1kr_pre=dw1kr_pre, dw1kr_con_arr=dw1kr_con_arr, dw1kr_pos_arr=dw1kr_pos_arr,
        duf_pre_arr=duf_pre_arr, duf_con_arr=duf_con_arr, duf_pos_arr=duf_pos_arr,
        her_pre_arr=her_pre_arr, her_con_arr=her_con_arr, her_pos_arr=her_pos_arr,
        shr_pre_arr=shr_pre_arr, shr_con_arr=shr_con_arr, shr_pos_arr=shr_pos_arr,
        fol_pre_arr=fol_pre_arr, fol_con_arr=fol_con_arr, fol_pos_arr=fol_pos_arr,
        bra_pre_arr=bra_pre_arr, bra_con_arr=bra_con_arr, bra_pos_arr=bra_pos_arr,
        mse_arr=mse_arr,
        duf_dep_pre_arr=duf_dep_pre_arr, duf_dep_con_arr=duf_dep_con_arr, duf_dep_pos_arr=duf_dep_pos_arr,
        fla_dur_arr=fla_dur_arr, smo_dur_arr=smo_dur_arr,
        fla_con_arr=fla_con_arr, smo_con_arr=smo_con_arr,
        lit_eq_arr=lit_eq_arr, duf_eq_arr=duf_eq_arr, herb_eq_arr=herb_eq_arr, shrub_eq_arr=shrub_eq_arr,
        burnup_adj_arr=burnup_adj_arr, burnup_err_arr=burnup_err_arr,
        lay0_arr=lay0_arr, lay2_arr=lay2_arr, lay4_arr=lay4_arr, lay6_arr=lay6_arr,
        lay60d_arr=lay60d_arr, lay275d_arr=lay275d_arr,
    )

