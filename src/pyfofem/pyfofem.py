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
from typing import Dict, List, Optional, Tuple, Union

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
    CONSUMPTION_VARS,
    SOIL_HEAT_VARS,
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

from .components.emission_calcs import (
    _EF_GROUP_DEFAULT,
    _EF_SMOLDERING_GROUP_DEFAULT,
    _EF_DUFF_GROUP_DEFAULT,
    _load_ef_csv,
    calc_smoke_emissions,
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
    :param ef_group: Emission factor group (1-8; default 3).
    :param ef_csv_path: Path to ``emissions_factors.csv`` override.
    :param units: Unit system. ``'Imperial'`` (T/ac, in) or ``'SI'``
        (kg/m², cm).
    :param moisture_regime: Optional named moisture regime. One of
        ``'wet'``, ``'moderate'``, ``'dry'``, or ``'very dry'``
        (case-insensitive). Overrides *duff_moist*, *dw10_moist*,
        *dw1000_moist*, and *l_moist* when provided.
    :param num_workers: Number of parallel workers for the burnup loop.
        ``1`` (default) runs sequentially. ``>1`` uses
        ``ProcessPoolExecutor``.
    :param show_progress: If ``True``, display a :mod:`tqdm` progress bar
        during the per-cell burnup loop. Default ``False``.
    :returns: Dict with all :data:`CONSUMPTION_VARS` keys.  Values are
        plain Python ``float``/``int`` when all inputs are scalars, otherwise
        ``np.ndarray``.
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
        for v in (region, cvr_grp, season, fuel_category)
    )
    scalar_call = _cat_scalar and all(_is_scalar(v) for v in _scalar_inputs)

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
    # 4. Vectorised sub-model outputs (litter, herb, shrub, canopy, mse, duff)
    # ------------------------------------------------------------------
    lit_con_arr = np.asarray(consm_litter(lit_a, l_m_a, cvr_grp=cvr_a, reg=reg_a, units=units), dtype=float)
    lit_pre_arr = lit_a.copy()
    lit_pos_arr = lit_pre_arr - lit_con_arr

    her_pre_arr = her_a.copy()
    her_con_arr = np.asarray(consm_herb(reg_a, cvr_a, lit_a, her_a, season=sea_a, units=units), dtype=float)
    her_pos_arr = her_pre_arr - her_con_arr

    shr_pre_arr = shr_a.copy()

    crown_res   = consm_canopy(pcb_a, fol_a, bra_a, units=units)
    fol_pre_arr = fol_a.copy()
    fol_con_arr = np.asarray(crown_res['flc'], dtype=float)
    fol_pos_arr = fol_pre_arr - fol_con_arr
    bra_pre_arr = bra_a.copy()
    bra_con_arr = np.asarray(crown_res['blc'], dtype=float)
    bra_pos_arr = bra_pre_arr - bra_con_arr

    mse_arr = np.clip(
        np.asarray(consm_mineral_soil(reg_a, cvr_a, ft_a, duf_m_a, 'edm'), dtype=float),
        0.0, 100.0,
    )

    duf_pre_arr     = duf_a.copy()
    duf_dep_pre_arr = duf_dep_a.copy()

    # Call consm_duff per-cell so region/cover-group routing is correct,
    # then stack results into arrays.
    pdc_list = np.empty(n, dtype=float)
    ddc_list = np.empty(n, dtype=float)
    rdd_list = np.empty(n, dtype=float)

    for _i in range(n):
        _pre_l110 = float(lit_a[_i] + dw10_a[_i] + dw1_a[_i])
        _pre_dl110 = float(_pre_l110 + duf_a[_i])
        _res = consm_duff(
            float(duf_a[_i]), float(duf_m_a[_i]),
            reg=str(reg_a[_i]) if reg_a.size > 0 else None,
            cvr_grp=str(cvr_a[_i]) if cvr_a.size > 0 else None,
            duff_moist_cat='edm',
            d_pre=float(duf_dep_a[_i]),
            dw1000_moist=float(dw1k_m_a[_i]),
            pre_l110=_pre_l110,
            pre_dl110=_pre_dl110,
            units=units,
        )
        pdc_list[_i] = float(np.asarray(_res['pdc']).ravel()[0])
        ddc_list[_i] = float(np.asarray(_res['ddc']).ravel()[0]) if _res['ddc'] is not None else np.nan
        rdd_list[_i] = float(np.asarray(_res['rdd']).ravel()[0]) if _res['rdd'] is not None else np.nan

    pdc_arr = np.clip(pdc_list, 0.0, 100.0)

    duf_dep_con_arr = np.clip(ddc_list, 0.0, duf_dep_pre_arr)
    duf_dep_pos_arr = duf_dep_pre_arr - duf_dep_con_arr

    # ------------------------------------------------------------------
    # 5. Per-cell burnup (parallelised)
    # ------------------------------------------------------------------
    slc_pct_arr = np.clip(
        np.asarray(consm_shrub(
            reg_a, cvr_a, shr_a, season=sea_a,
            pre_ll=lit_a,
            pre_dl=duf_a,
            pre_rl=np.zeros_like(shr_a),
            duff_moist=duf_m_a,
            llc=lit_con_arr,
            ddc=duf_a * pdc_arr / 100.0,
            units=units,
        ), dtype=float),
        0.0, 100.0,
    )
    # C++ Eq 234 parity for SouthEast non-Pocosin non-Flatwoods shrub.
    # The C++ route uses Equation_16 and Eq_234_Per directly from CI loads.
    _fw = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd')
    _is_se_np = (
        (reg_a == 'SouthEast') &
        ~np.isin(cvr_a, ('Pocosin', 'PC')) &
        ~np.isin(cvr_a, _fw)
    )
    if np.any(_is_se_np):
        _wpre = lit_a + duf_a + dw10_a + dw1_a
        _wpre_safe = np.maximum(_wpre, 1e-12)
        _eq16_w = 3.4958 + (0.3833 * _wpre) - (0.0237 * duf_m_a) - (5.6075 / _wpre_safe)
        _shr_safe = np.maximum(shr_pre_arr, 1e-12)
        _f = (
            (3.2484 + (0.4322 * _wpre) + (0.6765 * shr_pre_arr) -
             (0.0276 * duf_m_a) - (5.0796 / _wpre_safe) - _eq16_w) / _shr_safe
        )
        _f = np.where((_wpre <= 0.0) | (shr_pre_arr <= 0.0) | (_eq16_w == 0.0), 0.0, _f)
        _eq234_pct = np.clip(_f * 100.0, 0.0, 100.0)
        slc_pct_arr = np.where(_is_se_np, _eq234_pct, slc_pct_arr)

    shr_con_arr = shr_pre_arr * slc_pct_arr / 100.0
    shr_pos_arr = shr_pre_arr - shr_con_arr

    # Output arrays initialised to simplified-default values
    duf_con_arr   = duf_pre_arr * pdc_arr / 100.0
    duf_pos_arr   = duf_pre_arr - duf_con_arr

    dw1_pre_arr   = dw1_a.copy();   dw1_con_arr  = dw1_pre_arr.copy();   dw1_pos_arr  = np.zeros(n)
    dw10_pre_arr  = dw10_a.copy();  dw10_con_arr = dw10_pre_arr.copy();  dw10_pos_arr = np.zeros(n)
    pct100 = np.clip(0.95 - 0.008 * np.maximum(dw10_m_a - 10.0, 0.0), 0.50, 1.00)
    dw100_pre_arr = dw100_a.copy(); dw100_con_arr = dw100_pre_arr * pct100; dw100_pos_arr = dw100_pre_arr - dw100_con_arr
    pct1ks = np.clip(0.15 - 0.001 * dw1k_m_a, 0.02, 0.20)
    dw1ks_con_arr = dw1ks_pre * pct1ks;  dw1ks_pos_arr = dw1ks_pre - dw1ks_con_arr
    pct1kr = np.clip(0.45 - 0.003 * dw1k_m_a, 0.05, 0.50)
    dw1kr_con_arr = dw1kr_pre * pct1kr;  dw1kr_pos_arr = dw1kr_pre - dw1kr_con_arr

    snd_fla_arr = np.zeros(n);   snd_smo_arr = dw1ks_con_arr.copy()
    rot_fla_arr = np.zeros(n);   rot_smo_arr = dw1kr_con_arr.copy()
    fine_fla_arr = dw1_con_arr + dw10_con_arr + dw100_con_arr
    fine_smo_arr = np.zeros(n)
    lit_fla_arr  = lit_con_arr.copy();  lit_smo_arr = np.zeros(n)
    fla_dur_arr  = np.where(np.isnan(frt_a), np.nan, frt_a)
    smo_dur_arr  = np.full(n, np.nan)
    burnup_ran   = np.zeros(n, dtype=bool)
    burnup_adj_arr = np.zeros(n, dtype=int)
    burnup_err_arr = np.zeros(n, dtype=int)

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
        _tqdm_kw = dict(total=len(cell_kwargs), desc='Burnup', unit='cell',
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
                    pool.map(_run_burnup_cell, cell_kwargs),
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
    # 7. Flaming / smoldering totals
    # ------------------------------------------------------------------
    fla_con_arr = (lit_fla_arr + fine_fla_arr + her_con_arr + shr_con_arr
                   + fol_con_arr + bra_con_arr + snd_fla_arr + rot_fla_arr)
    smo_con_arr = (duf_con_arr + lit_smo_arr + fine_smo_arr
                   + snd_smo_arr + rot_smo_arr)

    # ------------------------------------------------------------------
    # 8. Equation number arrays (broadcast from scalar categoricals)
    # ------------------------------------------------------------------
    _fw = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd')
    lit_eq_arr = np.where(
        np.isin(cvr_a, _fw), 997,
        np.where(reg_a == 'SouthEast', 998, 999)
    )
    duf_eq_arr = np.select(
        [np.isin(reg_a, ('InteriorWest','PacificWest')) & np.isin(cvr_a, ('Ponderosa pine','PN','Ponderosa')),
         np.isin(reg_a, ('InteriorWest','PacificWest')),
         (reg_a == 'SouthEast') & np.isin(cvr_a, ('Pocosin','PC')),
         reg_a == 'SouthEast',
         reg_a == 'NorthEast'],
        [4, 2, 20, 16, 3],
        default=2,
    )
    herb_eq_arr = np.where(
        reg_a == 'SouthEast', 222,
        np.where(np.isin(cvr_a, ('Grass','GG','GrassGroup')), 221,
        np.where(np.isin(cvr_a, _fw), 223, 22))
    )
    shrub_eq_arr = np.where(
        reg_a == 'SouthEast', np.where(np.isin(cvr_a, ('Pocosin','PC')), 233, 234),
        np.where(np.isin(cvr_a, ('Sagebrush','SB')), 232,
        np.where(np.isin(cvr_a, _fw), 236,
        np.where(np.isin(cvr_a, ('Shrub','SG','ShrubGroup')), 231, 23)))
    )

    # ------------------------------------------------------------------
    # 9. Smoke emissions (vectorised call)
    # ------------------------------------------------------------------
    emissions = calc_smoke_emissions(
        flaming_load=fla_con_arr,
        smoldering_load=smo_con_arr,
        mode=em_mode,
        ef_group=ef_group,
        duff_load=duf_con_arr,
        ef_csv_path=ef_csv_path,
        units=units,
    )

    # ------------------------------------------------------------------
    # 10. Squeeze back to scalars when called with scalar inputs
    # ------------------------------------------------------------------
    def _out(arr):
        """Return scalar float if scalar_call, otherwise np.ndarray."""
        a = np.asarray(arr, dtype=float)
        return float(a[0]) if scalar_call else a

    def _out_int(arr):
        a = np.asarray(arr, dtype=int)
        return int(a[0]) if scalar_call else a

    return {
        'LitPre':  _out(lit_pre_arr),  'LitCon':  _out(lit_con_arr),  'LitPos':  _out(lit_pos_arr),
        'DW1Pre':  _out(dw1_pre_arr),  'DW1Con':  _out(dw1_con_arr),  'DW1Pos':  _out(dw1_pos_arr),
        'DW10Pre': _out(dw10_pre_arr), 'DW10Con': _out(dw10_con_arr), 'DW10Pos': _out(dw10_pos_arr),
        'DW100Pre':_out(dw100_pre_arr),'DW100Con':_out(dw100_con_arr),'DW100Pos':_out(dw100_pos_arr),
        'DW1kSndPre':_out(dw1ks_pre), 'DW1kSndCon':_out(dw1ks_con_arr),'DW1kSndPos':_out(dw1ks_pos_arr),
        'DW1kRotPre':_out(dw1kr_pre), 'DW1kRotCon':_out(dw1kr_con_arr),'DW1kRotPos':_out(dw1kr_pos_arr),
        'DufPre':  _out(duf_pre_arr),  'DufCon':  _out(duf_con_arr),  'DufPos':  _out(duf_pos_arr),
        'HerPre':  _out(her_pre_arr),  'HerCon':  _out(her_con_arr),  'HerPos':  _out(her_pos_arr),
        'ShrPre':  _out(shr_pre_arr),  'ShrCon':  _out(shr_con_arr),  'ShrPos':  _out(shr_pos_arr),
        'FolPre':  _out(fol_pre_arr),  'FolCon':  _out(fol_con_arr),  'FolPos':  _out(fol_pos_arr),
        'BraPre':  _out(bra_pre_arr),  'BraCon':  _out(bra_con_arr),  'BraPos':  _out(bra_pos_arr),
        'MSE':     _out(mse_arr),
        'DufDepPre':_out(duf_dep_pre_arr),'DufDepCon':_out(duf_dep_con_arr),'DufDepPos':_out(duf_dep_pos_arr),
        **{k: (_out(v) if isinstance(v, np.ndarray) else v) for k, v in emissions.items()},
        'FlaDur':  _out(fla_dur_arr),  'SmoDur':  _out(smo_dur_arr),
        'FlaCon':  _out(fla_con_arr),  'SmoCon':  _out(smo_con_arr),
        # 'Lay0': float('nan'), 'Lay2': float('nan'),
        # 'Lay4': float('nan'), 'Lay6': float('nan'),
        # 'Lay60d': float('nan'), 'Lay275d': float('nan'),
        'Lit-Equ':    _out_int(lit_eq_arr),
        'DufCon-Equ': _out_int(duf_eq_arr),
        'DufRed-Equ': _out_int(duf_eq_arr),
        'MSE-Equ': 10,
        'Herb-Equ':   _out_int(herb_eq_arr),
        'Shurb-Equ':  _out_int(shrub_eq_arr),
        'BurnupLimitAdj':  _out_int(burnup_adj_arr),
        'BurnupError':     _out_int(burnup_err_arr),
    }

