# -*- coding: utf-8 -*-
"""
consumption_calcs.py – FOFEM fuel consumption and related calculations.

Provides fuel consumption models for each fuel component, as well as helpers
for running the Burnup post-frontal combustion model and calculating carbon
loadings.

Functions:
    calc_carbon            – Convert fuel loadings to carbon loadings.
    consm_canopy           – Crown/canopy fuel consumption.
    consm_duff             – Duff fuel consumption.
    consm_herb             – Herbaceous fuel consumption.
    consm_litter           – Litter fuel consumption.
    consm_mineral_soil     – Mineral soil exposure.
    consm_shrub            – Shrub fuel consumption.
    _extract_burnup_consumption     – Internal helper to extract per-class consumption.
    _burnup_durations               – Internal helper to extract flaming/smoldering durations.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .burnup import (
    FuelParticle,
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    burnup as _burnup,
    _FIRE_BOUNDS,
    _FUEL_BOUNDS,
    _BURNUP_LIMIT_ADJUST,
    _BURNUP_LIMIT_ERROR,
)
from ._component_helpers import _is_scalar, _maybe_scalar, _to_str_arr


# ---------------------------------------------------------------------------
# Carbon Calculations
# ---------------------------------------------------------------------------

# Fuel component groups and their carbon conversion factors (Penman 2003;
# Smith & Heath 2002)
_CARBON_FACTOR_WOODY = 0.50   # down woody, herb, shrub, foliage, branch
_CARBON_FACTOR_DUFF  = 0.37   # duff and litter

_CARBON_WOODY_KEYS = frozenset({
    'dw1', 'dw10', 'dw100',
    'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    'herb', 'shrub', 'foliage', 'branch',
})
_CARBON_DUFF_KEYS = frozenset({'duff', 'litter'})


# ---------------------------------------------------------------------------
# Output variable lists
# ---------------------------------------------------------------------------

CONSUMPTION_VARS = [
    'LitPre', 'LitCon', 'LitPos', 'DW1Pre', 'DW1Con', 'DW1Pos', 'DW10Pre', 'DW10Con', 'DW10Pos',
    'DW100Pre', 'DW100Con', 'DW100Pos', 'DW1kSndPre', 'DW1kSndCon', 'DW1kSndPos',
    'DW1kRotPre', 'DW1kRotCon', 'DW1kRotPos', 'DufPre', 'DufCon', 'DufPos',
    'HerPre', 'HerCon', 'HerPos', 'ShrPre', 'ShrCon', 'ShrPos',
    'FolPre', 'FolCon', 'FolPos', 'BraPre', 'BraCon', 'BraPos',
    'MSE', 'DufDepPre', 'DufDepCon', 'DufDepPos',
    'PM10F', 'PM10S', 'PM25F', 'PM25S', 'CH4F', 'CH4S', 'COF', 'COS', 'CO2F', 'CO2S',
    'NOXF', 'NOXS', 'SO2F', 'SO2S',
    'PM10S_Duff', 'PM25S_Duff', 'CH4S_Duff', 'COS_Duff', 'CO2S_Duff', 'NOXS_Duff', 'SO2S_Duff',
    'FlaDur', 'SmoDur', 'FlaCon', 'SmoCon',
    'Lay0', 'Lay2', 'Lay4', 'Lay6', 'Lay60d', 'Lay275d',
    'Lit-Equ', 'DufCon-Equ', 'DufRed-Equ', 'MSE-Equ', 'Herb-Equ', 'Shrub-Equ',
    'BurnupLimitAdj', 'BurnupError'
]
SOIL_HEAT_VARS = ['Lay0', 'Lay2', 'Lay4', 'Lay6', 'Lay60d', 'Lay275d']


# ---------------------------------------------------------------------------
# Categorical parameter lookup tables (int code → string label)
# ---------------------------------------------------------------------------

#: Integer codes for FOFEM regions.
REGION_CODES: Dict[int, str] = {
    1: 'InteriorWest',
    2: 'PacificWest',
    3: 'NorthEast',
    4: 'SouthEast',
}

#: Integer codes for FOFEM cover groups.
CVR_GRP_CODES: Dict[int, str] = {
    0:  '',
    1:  'Ponderosa pine',
    2:  'Pocosin',
    3:  'Chaparral',
    4:  'Shrub-Chaparral',
    5:  'Sagebrush',
    6:  'Flatwood',
    7:  'Pine Flatwoods',
    8:  'Red Jack Pine',
    9:  'Red, Jack Pine',
    10: 'Grass',
    11: 'Shrub',
    # Short aliases
    12: 'PN',
    13: 'PC',
    14: 'SGC',
    15: 'ShrubGroupChaparral',
    16: 'SB',
    17: 'PFL',
    18: 'PinFltwd',
    19: 'RJP',
    20: 'RedJacPin',
    21: 'GG',
    22: 'GrassGroup',
    23: 'SG',
    24: 'ShrubGroup',
}

#: Integer codes for burn seasons.
SEASON_CODES: Dict[int, str] = {
    1: 'Spring',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter',
}

#: Integer codes for fuel categories.
FUEL_CATEGORY_CODES: Dict[int, str] = {
    1: 'Natural',
    2: 'Slash',
}


def _to_str_arr(
    val: Union[str, int, np.ndarray],
    lut: Dict[int, str],
) -> np.ndarray:
    """Convert a categorical parameter to a 1-D numpy string array.

    Accepts a plain string, an integer code (looked up in *lut*), or a
    numpy array of strings or integer codes.  Scalar inputs return a
    length-1 array; array inputs are returned as a 1-D string array.

    :param val: Input value — ``str``, ``int``, or ``np.ndarray``.
    :param lut: Integer → string lookup table (e.g. :data:`REGION_CODES`).
    :returns: 1-D ``np.ndarray`` of ``dtype=object`` (strings).
    :raises KeyError: If an integer code is not present in *lut*.
    """
    if isinstance(val, np.ndarray):
        flat = val.ravel()
        out = np.empty(flat.shape, dtype=object)
        for i, v in enumerate(flat):
            if isinstance(v, (int, np.integer)):
                out[i] = lut[int(v)]
            else:
                out[i] = str(v)
        return out
    if isinstance(val, (int, np.integer)):
        return np.array([lut[int(val)]], dtype=object)
    return np.array([str(val)], dtype=object)


# ---------------------------------------------------------------------------
# Moisture regime lookup
# ---------------------------------------------------------------------------

_MOISTURE_REGIMES: Dict[str, Dict[str, float]] = {
    'wet':      {'duff': 130.0, '10hr': 22.0, '3plus': 40.0, 'soil': 25.0},
    'moderate': {'duff':  75.0, '10hr': 16.0, '3plus': 30.0, 'soil': 15.0},
    'dry':      {'duff':  40.0, '10hr': 10.0, '3plus': 15.0, 'soil': 10.0},
    'very dry': {'duff':  20.0, '10hr':  6.0, '3plus': 10.0, 'soil':  5.0},
}


def get_moisture_regime(regime: str) -> Dict[str, float]:
    """
    Return default fuel moisture values (%) for a named FOFEM moisture regime.

    Four western moisture regimes are defined (Lutes 2020, p. 79):

    +----------+---------+---------+----------+---------+
    | Regime   | Duff    | 10-hr   | 3+ in.   | Soil    |
    +==========+=========+=========+==========+=========+
    | Wet      | 130 %   | 22 %    | 40 %     | 25 %    |
    | Moderate |  75 %   | 16 %    | 30 %     | 15 %    |
    | Dry      |  40 %   | 10 %    | 15 %     | 10 %    |
    | Very dry |  20 %   |  6 %    | 10 %     |  5 %    |
    +----------+---------+---------+----------+---------+

    :param regime: One of ``'wet'``, ``'moderate'``, ``'dry'``, or ``'very dry'``
        (case-insensitive).
    :returns: Dict with keys ``'duff'``, ``'10hr'``, ``'3plus'``, ``'soil'``
        and float values in percent.
    :raises KeyError: If *regime* is not one of the four recognised values.
    """
    key = regime.strip().lower()
    if key not in _MOISTURE_REGIMES:
        raise KeyError(
            f"Unknown moisture regime '{regime}'. "
            f"Valid options: {list(_MOISTURE_REGIMES.keys())}"
        )
    return dict(_MOISTURE_REGIMES[key])


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def calc_carbon(
    loadings: Dict[str, Union[float, np.ndarray]],
    units: str = 'SI',
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Convert fuel loadings to carbon loadings using FOFEM conversion factors.

    Two fixed factors are applied (Lutes 2020, p. 79):

    * Down woody, herbaceous, shrub, foliage, branch → Carbon = loading × 0.50
      (Penman et al. 2003)
    * Duff and litter → Carbon = loading × 0.37
      (Smith & Heath 2002)

    :param loadings: Dict of fuel-component loadings. Recognised keys:

        * ``'dw1'``, ``'dw10'``, ``'dw100'`` – timelag woody fuels
        * ``'dwk_3_6'``, ``'dwk_6_9'``, ``'dwk_9_20'``, ``'dwk_20'`` – 1000-hr fuels
        * ``'herb'``, ``'shrub'``, ``'foliage'``, ``'branch'``
        * ``'duff'``, ``'litter'``

        Values may be Python scalars or :class:`numpy.ndarray`.  Units are
        not converted — values are returned in the same units as supplied.
    :param units: ``'SI'`` (kg/m²) or ``'imperial'`` (T/ac).  Currently
        informational only; no unit conversion is performed.
    :returns: Dict with the same keys as *loadings* and carbon loading values
        in the same units.
    :raises ValueError: If *loadings* contains a key that is not recognised.
    """
    result: Dict[str, Union[float, np.ndarray]] = {}
    all_known = _CARBON_WOODY_KEYS | _CARBON_DUFF_KEYS
    for key, val in loadings.items():
        if key in _CARBON_WOODY_KEYS:
            result[key] = val * _CARBON_FACTOR_WOODY
        elif key in _CARBON_DUFF_KEYS:
            result[key] = val * _CARBON_FACTOR_DUFF
        else:
            raise ValueError(
                f"Unrecognised fuel component key '{key}'. "
                f"Known keys: {sorted(all_known)}"
            )
    return result


def consm_canopy(
    crown_burn: Union[float, np.ndarray],
    pre_fl: Union[float, np.ndarray],
    pre_bl: Union[float, np.ndarray],
    units: str = 'SI',
) -> dict:
    """
    FOFEM canopy (crown fire) fuel consumption model.

    Estimates foliage and branch load consumed during a crown fire, given a
    user-provided estimate of the proportion of the stand affected by crown
    fire.

    Accepts scalar or array inputs. Scalar values in the returned dict
    correspond to scalar numeric inputs; arrays are returned otherwise.

    :param crown_burn: Proportion of stand area affected by crown fire (%).
        Scalar or np.ndarray.
    :param pre_fl: Pre-fire foliage fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param pre_bl: Pre-fire branch fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Dict with keys:

        - ``'flc'`` – foliage load consumed (same units as input). Scalar
          ``float`` when all numeric inputs are scalars, otherwise
          ``np.ndarray``.
        - ``'blc'`` – branch load consumed (same units as input; 50% of the
          proportion consumed by crown fire). Scalar ``float`` when all
          numeric inputs are scalars, otherwise ``np.ndarray``.
    """
    scalar_input = _is_scalar(crown_burn) and _is_scalar(pre_fl) and _is_scalar(pre_bl)

    crown_burn = np.atleast_1d(np.asarray(crown_burn, dtype=float))
    pre_fl = np.atleast_1d(np.asarray(pre_fl, dtype=float))
    pre_bl = np.atleast_1d(np.asarray(pre_bl, dtype=float))

    if units == 'SI':
        pre_fl = pre_fl * 4.4609  # kg/m² → T/acre
        pre_bl = pre_bl * 4.4609

    flc = (crown_burn / 100) * pre_fl
    blc = (crown_burn / 100) * pre_bl * 0.5

    if units == 'SI':
        flc = flc / 4.4609  # T/acre → kg/m²
        blc = blc / 4.4609

    return {
        'flc': _maybe_scalar(flc, scalar_input),
        'blc': _maybe_scalar(blc, scalar_input),
    }


def consm_duff(
    pre_dl: Union[float, np.ndarray],
    duff_moist: Union[float, np.ndarray],
    reg: Optional[str] = None,
    cvr_grp: Optional[str] = None,
    duff_moist_cat: Optional[str] = None,
    d_pre: Optional[Union[float, np.ndarray]] = None,
    mc_lyr1: Optional[Union[float, np.ndarray]] = None,
    pre_dl110: Optional[Union[float, np.ndarray]] = None,
    pre_l110: Optional[Union[float, np.ndarray]] = None,
    dw1000_moist: Optional[Union[float, np.ndarray]] = None,
    pile: bool = False,
    units: str = 'SI',
) -> dict:
    """
    FOFEM duff consumption model.

    Mirrors the logic of ``DUF_Mngr`` / region sub-functions in the C++
    source (``fof_duf.cpp``).  Computes:

    * ``'pdc'`` – percent of duff load consumed (%).
    * ``'ddc'`` – duff depth consumed (same depth units as *d_pre*).
    * ``'rdd'`` – residual (post-fire) duff depth (same units as *d_pre*).

    **Depth reduction approach** – Per C++ Note-5 (``DUF_Mngr``, 2016), the
    raw regression-based depth-reduction equations (Eqs 5, 6, 7) are no
    longer used for normal runs.  Instead, depth consumed is derived directly
    from the percent consumed::

        ddc = d_pre × (pdc / 100)
        rdd = d_pre − ddc

    This matches the final line of ``DUF_Mngr``:
    ``a_DUF->f_Red = a_CI->f_DufDep * (a_DUF->f_Per / 100.0)``.

    **Percent consumed routing** (matches ``DUF_Mngr`` priority order):

    +---------------------+---------------+------------+--------+
    | Region              | Cover type    | Moist cat  | Eq     |
    +=====================+===============+============+========+
    | Any                 | Piles         | —          | 17     |
    +---------------------+---------------+------------+--------+
    | Any                 | Chaparral/SGC | —          | 19     |
    +---------------------+---------------+------------+--------+
    | InteriorWest /      | Ponderosa     | ldm        | 4      |
    | PacificWest         | Ponderosa     | edm        | 2      |
    |                     | Ponderosa     | nfdth      | 3      |
    |                     | Other         | ldm        | 1      |
    |                     | Other         | edm        | 2      |
    |                     | Other         | nfdth      | 3      |
    +---------------------+---------------+------------+--------+
    | NorthEast           | RedJacPine    | edm        | 15     |
    |                     | RedJacPine    | ldm/nfdth  | 2 (dflt|
    |                     | BalsamSpruce  | ldm        | 5→pct  |
    |                     | BalsamSpruce  | edm        | 15     |
    |                     | BalsamSpruce  | nfdth      | 3      |
    |                     | Other         | —          | 2 (dflt|
    +---------------------+---------------+------------+--------+
    | SouthEast           | Pocosin       | —          | 20     |
    |                     | Other         | —          | 16     |
    +---------------------+---------------+------------+--------+
    | Any / fallback      | —             | —          | 2      |
    +---------------------+---------------+------------+--------+

    :param pre_dl: Pre-fire duff load (Mg/ha if ``units='SI'``, T/acre if
        ``units='Imperial'``). Scalar or np.ndarray.
    :param duff_moist: Duff moisture content (%). Scalar or np.ndarray.
    :param reg: Region name. One of ``'InteriorWest'``, ``'PacificWest'``,
        ``'NorthEast'``, ``'SouthEast'``.
    :param cvr_grp: Cover group name (e.g. ``'Ponderosa pine'``,
        ``'Pocosin'``, ``'Chaparral'``).
    :param duff_moist_cat: Duff moisture category. One of:

        - ``'ldm'`` – lower duff moisture
        - ``'edm'`` – entire / average duff moisture
        - ``'nfdth'`` – NFDR 1,000-hour moisture content

    :param d_pre: Pre-fire duff depth (cm if ``units='SI'``, inches if
        ``units='Imperial'``). Required for ``'ddc'`` and ``'rdd'`` outputs.
    :param mc_lyr1: Surface-layer moisture content (%). Required for Eq 20
        (SouthEast Pocosin).
    :param pre_dl110: Pre-fire duff + litter + 10-hr load (same mass units as
        *pre_dl*). Required for Eq 16 (SouthEast non-Pocosin).
    :param pre_l110: Pre-fire litter + 10-hr load (same units). Required for
        Eq 16.
    :param dw1000_moist: 1000-hr fuel moisture content (%). Used by Eq 3 and
        the NorthEast Balsam/Spruce nfdth path.  Falls back to *duff_moist*
        when ``None``.
    :param pile: ``True`` for pile burning (Eq 17 – 10 % consumed).
    :param units: ``'SI'`` (default, kg/m² / cm) or ``'Imperial'`` (T/ac /
        in).

    :return: Dict with keys ``'pdc'``, ``'ddc'``, ``'rdd'``.
        ``'ddc'`` and ``'rdd'`` are ``None`` when *d_pre* is not supplied.
    """
    scalar_input = _is_scalar(pre_dl) and _is_scalar(duff_moist)

    pre_dl     = np.atleast_1d(np.asarray(pre_dl,     dtype=float))
    duff_moist = np.atleast_1d(np.asarray(duff_moist, dtype=float))
    if d_pre is not None:
        d_pre = np.atleast_1d(np.asarray(d_pre, dtype=float))
    if mc_lyr1 is not None:
        mc_lyr1 = np.atleast_1d(np.asarray(mc_lyr1, dtype=float))
    if pre_dl110 is not None:
        pre_dl110 = np.atleast_1d(np.asarray(pre_dl110, dtype=float))
    if pre_l110 is not None:
        pre_l110 = np.atleast_1d(np.asarray(pre_l110, dtype=float))

    # Eq 3 / NE-Balsam-nfdth use 1000-hr moisture (C++ f_MoistDW1000).
    # Fall back to duff_moist when not supplied.
    if dw1000_moist is not None:
        dw1k = np.atleast_1d(np.asarray(dw1000_moist, dtype=float))
    else:
        dw1k = duff_moist

    if units == 'SI':
        pre_dl = pre_dl * 4.4609                 # Mg/ha → T/acre
        if d_pre is not None:
            d_pre = d_pre / 2.54                 # cm → in
        if pre_dl110 is not None:
            pre_dl110 = pre_dl110 * 4.4609
        if pre_l110 is not None:
            pre_l110  = pre_l110  * 4.4609

    # ------------------------------------------------------------------
    # Convenience flag sets (matching C++ CI_is* predicates)
    # ------------------------------------------------------------------
    _IW_PW     = {'InteriorWest', 'PacificWest'}
    _PONDEROSA = {'Ponderosa pine', 'PN', 'Ponderosa'}
    _POCOSIN   = {'Pocosin', 'PC'}
    _CHAPARRAL = {'Chaparral', 'Shrub-Chaparral', 'SGC', 'ShrubGroupChaparral'}
    _REDJAC    = {'Red Jack Pine', 'Red, Jack Pine', 'RedJacPin', 'RJP'}
    _BALSAM    = {'Balsam', 'Black Spruce', 'Red Spruce', 'White Spruce',
                  'BalBRWSpr', 'Balsam Fir', 'BFS'}

    is_iw_pw    = reg in _IW_PW
    is_ne       = reg == 'NorthEast'
    is_se       = reg == 'SouthEast'
    is_ponderosa = cvr_grp in _PONDEROSA
    is_pocosin  = cvr_grp in _POCOSIN
    is_chaparral = cvr_grp in _CHAPARRAL
    is_redjac   = cvr_grp in _REDJAC
    is_balsam   = cvr_grp in _BALSAM

    # ------------------------------------------------------------------
    # pdc – percent consumed
    # Priority mirrors DUF_Mngr: Piles → Chaparral → region branches
    # ------------------------------------------------------------------
    pdc = np.full_like(duff_moist, np.nan)

    if pile:
        # Eq 17 – pile burning: 10 %
        pdc = np.full_like(duff_moist, 10.0)

    elif is_chaparral:
        # Eq 19 – Chaparral/SGC: 100 %
        pdc = np.full_like(duff_moist, 100.0)

    elif is_iw_pw:
        # PacificWest Slash → same as InteriorWest (C++ DUF_PacificWest Note-1)
        if is_ponderosa:
            if duff_moist_cat == 'ldm':
                pdc = 89.9 - 0.55 * duff_moist                  # Eq 4
            elif duff_moist_cat == 'edm':
                pdc = 83.7 - 0.426 * duff_moist                 # Eq 2
            else:                                                 # nfdth
                pdc = 114.7 - 4.2 * dw1k                        # Eq 3
        else:
            if duff_moist_cat == 'ldm':
                pdc = np.where(duff_moist <= 160,
                               97.1 - 0.519 * duff_moist, 13.6) # Eq 1
            elif duff_moist_cat == 'edm':
                pdc = 83.7 - 0.426 * duff_moist                 # Eq 2
            else:                                                 # nfdth
                pdc = 114.7 - 4.2 * dw1k                        # Eq 3

    elif is_ne:
        if is_redjac:
            if duff_moist_cat == 'edm':
                # Eq 15 with pine=1: derive pdc from residual depth
                f_rdd = (-0.791 + 0.004 * duff_moist
                         + 0.8 * (d_pre if d_pre is not None
                                  else np.zeros_like(duff_moist))
                         + 0.56)
                f_red = np.clip(
                    (d_pre if d_pre is not None
                     else np.zeros_like(duff_moist)) - f_rdd,
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            else:
                # ldm or nfdth → default (Eq 2)
                pdc = 83.7 - 0.426 * duff_moist
        elif is_balsam:
            if duff_moist_cat == 'ldm':
                # Eq 5 → derive pdc from depth reduction
                f_red_5 = np.clip(
                    1.028 - 0.0089 * duff_moist
                    + 0.417 * (d_pre if d_pre is not None
                               else np.zeros_like(duff_moist)),
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red_5 / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            elif duff_moist_cat == 'edm':
                # Eq 15 with pine=0
                f_rdd = (-0.791 + 0.004 * duff_moist
                         + 0.8 * (d_pre if d_pre is not None
                                  else np.zeros_like(duff_moist)))
                f_red = np.clip(
                    (d_pre if d_pre is not None
                     else np.zeros_like(duff_moist)) - f_rdd,
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            else:
                # nfdth → Eq 3
                pdc = 114.7 - 4.2 * dw1k
        else:
            if duff_moist_cat == 'edm':
                # Eq 15 with pine=0 for NorthEast non-RedJac, non-Balsam.
                f_rdd = (-0.791 + 0.004 * duff_moist
                         + 0.8 * (d_pre if d_pre is not None
                                  else np.zeros_like(duff_moist)))
                f_red = np.clip(
                    (d_pre if d_pre is not None
                     else np.zeros_like(duff_moist)) - f_rdd,
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            else:
                # NorthEast default (Duf_Default) → Eq 2
                pdc = 83.7 - 0.426 * duff_moist

    elif is_se:
        if is_pocosin:
            # Eq 20 – Pocosin per-layer load-based algorithm (C++ Equ_20_PerRed_Pocosin)
            # Works on duff load per 4-inch layer; mc_lyr1 is the top-layer moisture.
            mc0    = float(mc_lyr1[0]) if mc_lyr1 is not None else float(duff_moist[0])
            dl_val = float(pre_dl[0])
            dp_val = float(d_pre[0]) if d_pre is not None else 0.0

            if dp_val > 0 and dl_val > 0:
                f_10th        = dl_val / (dp_val * 10.0)  # load per 0.1-inch slice
                layer_load    = f_10th * 40.0 if dp_val >= 4.0 else dl_val  # per 4-in layer
                dep_rem       = dp_val
                duf_rem       = dl_val
                mc_layer      = mc0
                moi_inc       = 0.0
                tot_consumed  = 0.0
                _E_MINERAL    = 5.0

                while True:
                    cur_layer_load = layer_load if dep_rem >= 4.0 else duf_rem
                    if mc_layer < 10.0:
                        f_per = 1.0
                    elif mc_layer < 30.0:
                        f_per = 1.0 - (mc_layer * 0.00167)
                    elif mc_layer < 140.0:
                        f_per = 1.0 / (1.0 + np.exp(-1.0 * (
                            2.033 - (0.043 * mc_layer) + (0.44 * _E_MINERAL))))
                    elif mc_layer < 170.0:
                        f_per = 0.143441 - ((mc_layer - 140.0) * 0.0049)
                    else:
                        f_per = 0.0
                    f_per = max(f_per, 0.0)
                    tot_consumed += cur_layer_load * f_per

                    if dep_rem < 4.0:
                        break
                    dep_rem -= 4.0
                    duf_rem -= layer_load
                    if dep_rem < 4.0:
                        layer_load = duf_rem
                    moi_inc = min(moi_inc + 3.0, 12.0)
                    mc_layer += moi_inc

                pdc_val = np.clip((tot_consumed / dl_val) * 100.0, 0.0, 100.0)
            else:
                pdc_val = 0.0
            pdc = np.atleast_1d(np.full_like(duff_moist, pdc_val))

        else:
            # SE non-Pocosin – Eq 16
            # f_WPRE = lit + duff + dw10 + dw1  (here approximated by pre_dl110)
            # f_L    = lit + dw10 + dw1          (here approximated by pre_l110)
            if pre_dl110 is not None and pre_l110 is not None:
                f_wpre = np.where(pre_dl110 > 0, pre_dl110, 0.0)
                f_w = np.where(
                    f_wpre > 0,
                    3.4958 + (0.3833 * f_wpre) - (0.0237 * duff_moist) - (5.6075 / np.maximum(f_wpre, 1e-12)),
                    0.0,
                )
                f_l = pre_l110
                duff_only = f_wpre - f_l
                pdc = np.where(
                    f_w <= f_l, 0.0,
                    np.where(
                        duff_only > 0,
                        np.clip(100.0 * (f_w - f_l) / duff_only, 0.0, 100.0),
                        0.0,
                    ),
                )
            else:
                # Fall back to Eq 2 if required inputs missing
                pdc = 83.7 - 0.426 * duff_moist

    else:
        # Global fallback → Duf_Default → Eq 2
        pdc = 83.7 - 0.426 * duff_moist

    # C++ DUF_Mngr: duff_moist ≤ 10 forces 100 % consumed (Note, 2012)
    low_moist_mask = duff_moist <= 10.0
    pdc = np.where(low_moist_mask, 100.0, pdc)

    # Clamp to [0, 100] (C++ DUF_Mngr Note-1)
    pdc = np.clip(pdc, 0.0, 100.0)

    # ------------------------------------------------------------------
    # ddc / rdd – depth outputs.
    # Match golden/C++ equation-level behavior:
    #   Eq5: ddc = 1.028 - 0.0089*moist + 0.417*d_pre
    #   Eq6: ddc = 0.8811 - 0.0096*moist + 0.439*d_pre
    #   Eq7: ddc = 1.773 - 0.1051*dw1000_moist + 0.399*d_pre
    #   Eq15: rdd = -0.791 + 0.004*moist + 0.8*d_pre + 0.56*pine
    # Fall back to pdc-derived depth when no specific depth equation applies.
    # ------------------------------------------------------------------
    ddc = None
    rdd = None
    if d_pre is not None:
        d_pre_in = d_pre  # already in inches if units=='SI' was converted above
        if is_iw_pw:
            if duff_moist_cat == 'ldm':
                ddc = 1.028 - 0.0089 * duff_moist + 0.417 * d_pre_in  # Eq 5
            elif duff_moist_cat == 'edm':
                ddc = 0.8811 - 0.0096 * duff_moist + 0.439 * d_pre_in  # Eq 6
            elif duff_moist_cat == 'nfdth':
                ddc = 1.773 - 0.1051 * dw1k + 0.399 * d_pre_in  # Eq 7
            else:
                ddc = d_pre_in * (pdc / 100.0)
            ddc = np.clip(ddc, 0.0, d_pre_in)
            rdd = d_pre_in - ddc
        elif is_ne and duff_moist_cat == 'edm':
            pine_flag = 1.0 if is_redjac else 0.0
            rdd = -0.791 + 0.004 * duff_moist + 0.8 * d_pre_in + 0.56 * pine_flag  # Eq 15
            rdd = np.clip(rdd, 0.0, d_pre_in)
            ddc = d_pre_in - rdd
        else:
            ddc = np.clip(d_pre_in * (pdc / 100.0), 0.0, d_pre_in)
            rdd = d_pre_in - ddc

        # Convert back to cm for SI callers
        if units == 'SI':
            ddc = ddc * 2.54
            rdd = rdd * 2.54

    return {
        'pdc': _maybe_scalar(pdc, scalar_input),
        'ddc': _maybe_scalar(ddc, scalar_input),
        'rdd': _maybe_scalar(rdd, scalar_input),
    }


def consm_herb(
    reg: Union[str, int, np.ndarray],
    cvr_grp: Union[str, int, np.ndarray],
    pre_ll: Union[float, np.ndarray],
    pre_hl: Union[float, np.ndarray],
    season: Union[str, int, np.ndarray, None] = None,
    units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM herbaceous fuel consumption model.

    Accepts scalar or array inputs for all parameters, including *reg*,
    *cvr_grp*, and *season* (strings, integer codes, or arrays thereof).

    :param reg: Region name or integer code (see :data:`REGION_CODES`).
    :param cvr_grp: Cover group name or integer code (see
        :data:`CVR_GRP_CODES`).
    :param pre_ll: Pre-fire litter fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param pre_hl: Pre-fire herbaceous fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param season: Burn season (``'Spring'``, ``'Summer'``, ``'Fall'``,
        ``'Winter'``) or integer code (see :data:`SEASON_CODES`). Only
        relevant for GrassGroup: Eq 221 (10%) applies in Spring; all other
        seasons use Eq 22 (100%). Optional; defaults to non-Spring behaviour.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Herbaceous load consumed (same units as input). Scalar ``float``
        when all numeric inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(pre_hl)

    pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
    pre_hl = np.atleast_1d(np.asarray(pre_hl, dtype=float))
    n = max(len(pre_ll), len(pre_hl))

    if units == 'SI':
        pre_ll = pre_ll * 4.4609
        pre_hl = pre_hl * 4.4609

    reg_arr  = _to_str_arr(reg, REGION_CODES)
    cvr_arr  = _to_str_arr(cvr_grp, CVR_GRP_CODES)
    sea_arr  = _to_str_arr(season if season is not None else '', SEASON_CODES)
    reg_arr  = np.broadcast_to(reg_arr,  (n,)) if reg_arr.size  == 1 else reg_arr
    cvr_arr  = np.broadcast_to(cvr_arr,  (n,)) if cvr_arr.size  == 1 else cvr_arr
    sea_arr  = np.broadcast_to(sea_arr,  (n,)) if sea_arr.size  == 1 else sea_arr

    _flatwood_vals = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd')
    _grass_vals    = ('Grass', 'GG', 'GrassGroup')

    is_se         = reg_arr == 'SouthEast'
    is_grass_spr  = np.isin(cvr_arr, _grass_vals) & (sea_arr == 'Spring')
    is_flatwood   = np.isin(cvr_arr, _flatwood_vals)

    hlc = np.select(
        [is_se, is_grass_spr, is_flatwood],
        [
            # Eq 222
            -0.059 + (0.004 * pre_ll) + (0.917 * pre_hl),
            # Eq 221 – 10% in Spring only
            pre_hl * 0.1,
            # Eq 223
            ((pre_hl * 2.24) * 0.9944) / 2.24,
        ],
        default=pre_hl.copy(),  # Eq 22 – 100%
    )

    if units == 'SI':
        hlc = hlc / 4.4609

    return float(hlc[0]) if scalar_input else hlc


def consm_litter(
    pre_ll: Union[float, np.ndarray],
    l_moist: Union[float, np.ndarray],
    cvr_grp: Union[str, int, np.ndarray, None] = None,
    reg: Union[str, int, np.ndarray, None] = None,
    units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM litter consumption model (Eqs 997–999).

    Accepts scalar or array inputs for all parameters, including *cvr_grp*
    and *reg* (which may be strings, integer codes, or arrays thereof).

    .. note::
        Most fuel consumption is simulated using Burnup. This function covers
        litter-specific override equations for Flatwoods and Southeast regions.

    :param pre_ll: Pre-fire litter load (Mg/ha if ``units='SI'``, T/acre if
        ``units='Imperial'``). Scalar or np.ndarray.
    :param l_moist: Litter moisture content (%). Scalar or np.ndarray.
    :param cvr_grp: Cover group name or integer code (see
        :data:`CVR_GRP_CODES`). Scalar or np.ndarray. Optional.
    :param reg: Region name or integer code (see :data:`REGION_CODES`).
        Scalar or np.ndarray. Optional.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Litter load consumed (kg/m² for ``'SI'``, T/acre for
        ``'Imperial'``). Scalar ``float`` when all numeric inputs are scalars,
        otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(l_moist)

    pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
    l_moist = np.atleast_1d(np.asarray(l_moist, dtype=float))
    n = max(len(pre_ll), len(l_moist))

    if units == 'SI':
        pre_ll = pre_ll * 4.4609  # kg/m² → T/acre

    # Resolve categorical strings to broadcast-compatible arrays
    cvr_arr = _to_str_arr(cvr_grp if cvr_grp is not None else '', CVR_GRP_CODES)
    reg_arr = _to_str_arr(reg if reg is not None else '', REGION_CODES)
    # Broadcast to n
    cvr_arr = np.broadcast_to(cvr_arr, (n,)) if cvr_arr.size == 1 else cvr_arr
    reg_arr = np.broadcast_to(reg_arr, (n,)) if reg_arr.size == 1 else reg_arr

    _flatwood_vals = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd')
    is_flatwood = np.isin(cvr_arr, _flatwood_vals)
    is_southeast = reg_arr == 'SouthEast'

    llc = np.select(
        [is_flatwood, is_southeast],
        [
            # Eq 997
            np.power(0.2871 + (0.9140 * np.sqrt(pre_ll)) - (0.0101 * l_moist), 2),
            # Eq 998
            pre_ll * 0.8,
        ],
        default=pre_ll.copy(),  # Eq 999
    )

    if units == 'SI':
        llc = llc / 4.4609  # T/acre → kg/m²

    return float(llc[0]) if scalar_input else llc


def consm_mineral_soil(
    reg: Union[str, int, np.ndarray],
    cvr_grp: Union[str, int, np.ndarray],
    fuel_type: Union[str, int, np.ndarray],
    duff_moist: Union[float, np.ndarray],
    duff_moist_cat: str,
    pile: bool = False,
    pdr: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM mineral soil exposure model.

    Accepts scalar or array inputs, including *reg*, *cvr_grp*, and
    *fuel_type* (strings, integer codes, or arrays thereof).

    :param reg: Region name or integer code (see :data:`REGION_CODES`).
    :param cvr_grp: Cover group name or integer code (see :data:`CVR_GRP_CODES`).
    :param fuel_type: Fuel type — ``'natural'`` / ``1`` or ``'slash'`` / ``2``
        (see :data:`FUEL_CATEGORY_CODES`).
    :param duff_moist: Duff moisture content (%). Scalar or np.ndarray.
    :param duff_moist_cat: Duff moisture category (``'ldm'``, ``'edm'``,
        ``'nfdth'``, or ``'%dr'``).
    :param pile: ``True`` for pile burning (returns 10%). Default ``False``.
    :param pdr: Percent duff reduction (%), required when
        ``duff_moist_cat='%dr'``. Scalar or np.ndarray.

    :return: Mineral soil exposure (%). Scalar ``float`` when all numeric
        inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(duff_moist)

    duff_moist = np.atleast_1d(np.asarray(duff_moist, dtype=float))
    n = len(duff_moist)
    if pdr is not None:
        pdr = np.atleast_1d(np.asarray(pdr, dtype=float))

    reg_arr = _to_str_arr(reg, REGION_CODES)
    cvr_arr = _to_str_arr(cvr_grp, CVR_GRP_CODES)
    ft_arr  = _to_str_arr(fuel_type, FUEL_CATEGORY_CODES)
    reg_arr = np.broadcast_to(reg_arr, (n,)) if reg_arr.size == 1 else reg_arr
    cvr_arr = np.broadcast_to(cvr_arr, (n,)) if cvr_arr.size == 1 else cvr_arr
    ft_arr  = np.broadcast_to(ft_arr,  (n,)) if ft_arr.size  == 1 else ft_arr

    ft_lower = np.array([v.lower() for v in ft_arr], dtype=object)

    mse = np.full(n, np.nan)

    if pile:
        mse[:] = 10.0
    else:
        is_iw_pw    = np.isin(reg_arr, ('InteriorWest', 'PacificWest'))
        is_pocosin  = np.isin(cvr_arr, ('Pocosin', 'PC'))
        is_slash    = ft_lower == 'slash'
        is_natural  = ft_lower == 'natural'

        pdr_vals = pdr if pdr is not None else np.zeros(n)

        mse = np.select(
            [
                is_iw_pw & is_slash   & (duff_moist_cat == 'ldm'),   # Eq 9
                is_iw_pw & is_natural & (duff_moist_cat == 'ldm'),   # Eq 13
                is_iw_pw & is_slash   & (duff_moist_cat == 'nfdth'), # Eq 11
                is_iw_pw & is_natural & (duff_moist_cat == 'nfdth'), # Eq 12
                is_iw_pw              & (duff_moist_cat == 'edm'),   # Eq 10
                is_pocosin,                                           # Eq 202
                ~is_iw_pw & ~is_pocosin & (duff_moist_cat == 'edm'),         # Eq 10
                ~is_iw_pw & ~is_pocosin & (duff_moist_cat == '%dr'),         # Eq 14
            ],
            [
                np.where(duff_moist <= 135,
                         80 - 0.507 * duff_moist,
                         23.5 - 0.0914 * duff_moist),
                60.4 - 0.440 * duff_moist,
                93.3 - 3.55  * duff_moist,
                94.3 - 4.96  * duff_moist,
                167.4 - 31.6 * np.log(duff_moist),
                np.zeros(n),
                167.4 - 31.6 * np.log(duff_moist),
                -8.98 + 0.899 * pdr_vals,
            ],
            default=np.full(n, np.nan),
        )

    return float(mse[0]) if scalar_input else mse


def consm_shrub(
    reg: Union[str, int, np.ndarray],
    cvr_grp: Union[str, int, np.ndarray],
    pre_sl: Union[float, np.ndarray],
    season: Union[str, int, np.ndarray, None] = None,
    pre_ll: Optional[Union[float, np.ndarray]] = None,
    pre_dl: Optional[Union[float, np.ndarray]] = None,
    pre_rl: Optional[Union[float, np.ndarray]] = None,
    duff_moist: Optional[Union[float, np.ndarray]] = None,
    llc: Optional[Union[float, np.ndarray]] = None,
    ddc: Optional[Union[float, np.ndarray]] = None,
    units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM shrub fuel consumption model.

    Accepts scalar or array inputs, including *reg*, *cvr_grp*, and *season*
    (strings, integer codes, or arrays thereof).

    :param reg: Region name or integer code (see :data:`REGION_CODES`).
    :param cvr_grp: Cover group name or integer code (see :data:`CVR_GRP_CODES`).
    :param pre_sl: Pre-fire shrub fuel load. Scalar or np.ndarray.
    :param season: Burn season or integer code (see :data:`SEASON_CODES`).
    :param pre_ll: Pre-fire litter load (SE non-Pocosin Eq 234). Optional.
    :param pre_dl: Pre-fire duff load (SE non-Pocosin Eq 234). Optional.
    :param pre_rl: Pre-fire regeneration load (SE non-Pocosin Eq 234). Optional.
    :param duff_moist: Duff moisture content (%). Optional.
    :param llc: Litter load consumed (SE non-Pocosin Eq 234). Optional.
    :param ddc: Duff depth consumed (SE non-Pocosin Eq 234). Optional.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Percent shrub load consumed (%). Scalar ``float`` when all
        numeric inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_sl)

    pre_sl = np.atleast_1d(np.asarray(pre_sl, dtype=float))
    n = len(pre_sl)

    if units == 'SI':
        pre_sl = pre_sl * 4.4609
        if pre_ll is not None:
            pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float)) * 4.4609
        if pre_dl is not None:
            pre_dl = np.atleast_1d(np.asarray(pre_dl, dtype=float)) * 4.4609
        if pre_rl is not None:
            pre_rl = np.atleast_1d(np.asarray(pre_rl, dtype=float)) * 4.4609
    else:
        if pre_ll is not None:
            pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
        if pre_dl is not None:
            pre_dl = np.atleast_1d(np.asarray(pre_dl, dtype=float))
        if pre_rl is not None:
            pre_rl = np.atleast_1d(np.asarray(pre_rl, dtype=float))

    if duff_moist is not None:
        duff_moist = np.atleast_1d(np.asarray(duff_moist, dtype=float))
    if llc is not None:
        llc = np.atleast_1d(np.asarray(llc, dtype=float))
    if ddc is not None:
        ddc = np.atleast_1d(np.asarray(ddc, dtype=float))

    reg_arr = _to_str_arr(reg, REGION_CODES)
    cvr_arr = _to_str_arr(cvr_grp, CVR_GRP_CODES)
    sea_arr = _to_str_arr(season if season is not None else '', SEASON_CODES)
    reg_arr = np.broadcast_to(reg_arr, (n,)) if reg_arr.size == 1 else reg_arr
    cvr_arr = np.broadcast_to(cvr_arr, (n,)) if cvr_arr.size == 1 else cvr_arr
    sea_arr = np.broadcast_to(sea_arr, (n,)) if sea_arr.size == 1 else sea_arr

    _flatwood_vals = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd')

    is_se       = reg_arr == 'SouthEast'
    is_pocosin  = np.isin(cvr_arr, ('Pocosin', 'PC'))
    is_sage     = np.isin(cvr_arr, ('Sagebrush', 'SB'))
    is_flatwood = np.isin(cvr_arr, _flatwood_vals)
    is_shrubgrp = np.isin(cvr_arr, ('Shrub', 'SG', 'ShrubGroup'))

    # SE Pocosin seasonal
    sea_spr_win = np.isin(sea_arr, ('Spring', 'Winter'))
    sea_sum_fal = np.isin(sea_arr, ('Summer', 'Fall'))
    sea_fall    = sea_arr == 'Fall'
    sea_spr_sum = np.isin(sea_arr, ('Spring', 'Summer'))

    # Eq 234 – SE non-Pocosin (requires optional params; fall back to nan)
    if all(x is not None for x in (pre_ll, pre_dl, pre_rl, duff_moist, llc, ddc)):
        combo = pre_ll + pre_dl
        combo_safe = np.where(combo > 0, combo, np.nan)
        denom_safe = np.where((pre_sl + pre_rl) > 0, (pre_sl + pre_rl), np.nan)
        eq234 = (((3.2484 + (0.4322 * combo) + (0.6765 * (pre_sl + pre_rl)) -
                   (0.0276 * duff_moist) - (5.0796 / combo_safe)) -
                  (llc + ddc)) / denom_safe) * 100
    else:
        eq234 = np.full(n, np.nan)

    # Eq 236 – Flatwood
    season_flag = np.where(sea_spr_sum, 1.0, 0.0)
    eq236 = -0.1889 + (0.9049 * np.log(np.maximum(pre_sl, 1e-12))) + (0.0676 * season_flag)

    slc = np.select(
        [
            is_se & is_pocosin & sea_spr_win,        # Eq 233
            is_se & is_pocosin & sea_sum_fal,        # Eq 235
            is_se & ~is_pocosin,                     # Eq 234
            is_sage & sea_fall,                      # Eq 233
            is_sage & ~sea_fall,                     # Eq 232
            is_flatwood,                             # Eq 236
            is_shrubgrp,                             # Eq 231
        ],
        [
            np.full(n, 90.0),
            np.full(n, 80.0),
            eq234,
            np.full(n, 90.0),
            np.full(n, 50.0),
            eq236,
            np.full(n, 80.0),
        ],
        default=np.full(n, 60.0),  # Eq 23
    )

    return float(slc[0]) if scalar_input else slc


# Removed burnup-related constants and functions (now in burnup_calcs.py)
